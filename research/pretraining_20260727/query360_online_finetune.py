#!/usr/bin/env python3
"""Short end-to-end 360 m TransientQuery initialization screen.

This runner complements the frozen-feature experiment by genuinely updating
the complete Panopticon backbone.  Within one initialization condition,
``current_only``, ``transient_query``, and
``scale_aware_transient_query`` start from byte-identical backbone and head
states, see the same row order, encode the same valid frames, and take the same
number of optimizer steps.  The only intervention is which historical
features are visible to the current-query attention mask.

Only explicit train and inner-validation manifests are accepted.  Any path
component containing test/sealed/holdout is refused by the shared data module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from query360_data import (  # noqa: E402
    DEFAULT_WV3_SRF,
    SENSOR_ORDER,
    Query360Dataset,
    StrictHashedFileCache,
    assert_safe_path,
    query360_collate,
    sha256_file,
)
from query360_model import (  # noqa: E402
    TransientQuery360Head,
    model_parameter_signature,
    state_dict_sha256,
    stratified_binary_metrics,
    transient_query_loss,
)
from src.utils.training import _load_backbone  # noqa: E402


SCRIPT_VERSION = "query360-online-finetune-v1"
ARMS = (
    "current_only",
    "transient_query",
    "scale_aware_transient_query",
)
DEFAULT_WEIGHTS = str(REPO_ROOT / "weights" / "panopticon_vitb14_teacher.pth")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def atomic_json_write(path: Path, value: Any) -> None:
    path = assert_safe_path(path, purpose="JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path = assert_safe_path(path, purpose="CSV output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    path = assert_safe_path(path, purpose="checkpoint output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def configure_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class EpochOrderSampler(Sampler[int]):
    """Version-stable epoch order independent of DataLoader worker RNG."""

    def __init__(self, rows: int, *, seed: int, shuffle: bool):
        self.rows = int(rows)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def order(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.rows, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.rows, generator=generator)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order().tolist())

    def __len__(self) -> int:
        return self.rows

    def order_sha256(self) -> str:
        order = self.order().contiguous()
        return hashlib.sha256(order.numpy().tobytes()).hexdigest()


class OnlineTransientQueryClassifier(nn.Module):
    """Native-channel Panopticon encoding followed by a hierarchical TQ head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        model_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        feature_dim = int(getattr(backbone, "embed_dim", 768))
        self.head = TransientQuery360Head(
            feature_dim=feature_dim,
            num_sensors=len(SENSOR_ORDER),
            num_roles=3,
            model_dim=model_dim,
            num_heads=num_heads,
            depth=2,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.feature_dim = feature_dim

    def encode_batch(
        self,
        batch: Mapping[str, Any],
        *,
        device: torch.device,
        amp_dtype: str,
        encoder_microbatch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Encode every declared valid frame and differentiably scatter it."""

        global_indices = batch["index"].long().tolist()
        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(global_indices)
        }
        if len(global_to_local) != len(global_indices):
            raise RuntimeError("Online batch contains duplicate global row indices.")
        encoded_parts: list[torch.Tensor] = []
        flat_index_parts: list[torch.Tensor] = []
        observations = 0
        for sensor_index, sensor_name in enumerate(SENSOR_ORDER):
            sensor_batch = batch["sensor_batches"][sensor_name]
            images = sensor_batch["images"]
            if images.shape[0] == 0:
                continue
            local_rows = torch.tensor(
                [
                    global_to_local[int(value)]
                    for value in sensor_batch["rows"].long().tolist()
                ],
                dtype=torch.long,
                device=device,
            )
            roles = sensor_batch["roles"].long().to(device)
            channel_ids = sensor_batch["channel_ids"].reshape(-1)
            if images.shape[1] != channel_ids.numel():
                raise RuntimeError(f"{sensor_name}: channel-id mismatch.")
            sensor_encoded: list[torch.Tensor] = []
            for start in range(0, len(images), encoder_microbatch):
                stop = min(start + encoder_microbatch, len(images))
                image_chunk = images[start:stop].to(device, non_blocking=True)
                channel_chunk = (
                    channel_ids.view(1, -1)
                    .expand(stop - start, -1)
                    .clone()
                    .to(device, non_blocking=True)
                )
                with autocast_context(device, amp_dtype):
                    output = self.backbone.forward_features(
                        {"imgs": image_chunk, "chn_ids": channel_chunk}
                    )
                    sensor_encoded.append(output["x_norm_clstoken"])
            encoded = torch.cat(sensor_encoded, dim=0)
            if encoded.shape[0] != local_rows.numel():
                raise RuntimeError(f"{sensor_name}: encoded observation count mismatch.")
            flat_indices = (
                local_rows * (len(SENSOR_ORDER) * 3)
                + int(sensor_index) * 3
                + roles
            )
            encoded_parts.append(encoded)
            flat_index_parts.append(flat_indices)
            observations += int(encoded.shape[0])

        if not encoded_parts:
            raise RuntimeError("Online batch has no valid sensor observations.")
        encoded_all = torch.cat(encoded_parts, dim=0)
        flat_indices = torch.cat(flat_index_parts, dim=0)
        batch_size = len(global_indices)
        dense_flat = encoded_all.new_zeros(
            (batch_size * len(SENSOR_ORDER) * 3, self.feature_dim)
        )
        # index_add is differentiable with respect to encoded_all.  Each valid
        # slot is unique; duplicate roles were already removed by the dataset.
        dense_flat = dense_flat.index_add(0, flat_indices, encoded_all)
        dense = dense_flat.view(
            batch_size, len(SENSOR_ORDER), 3, self.feature_dim
        )
        valid_mask = batch["valid_mask"].bool().to(device)
        declared = valid_mask.reshape(-1)
        emitted = torch.zeros_like(declared)
        emitted[flat_indices] = True
        if not torch.equal(declared, emitted):
            missing = torch.nonzero(declared ^ emitted, as_tuple=False).flatten()
            raise RuntimeError(
                f"Online encoded/declaration mask mismatch: {missing.tolist()[:20]}"
            )
        return dense, valid_mask, observations

    def forward_batch(
        self,
        batch: Mapping[str, Any],
        *,
        arm: str,
        device: torch.device,
        amp_dtype: str,
        encoder_microbatch: int,
    ):
        features, valid_mask, observations = self.encode_batch(
            batch,
            device=device,
            amp_dtype=amp_dtype,
            encoder_microbatch=encoder_microbatch,
        )
        with autocast_context(device, amp_dtype):
            output = self.head(features, valid_mask, arm=arm)
        return output, valid_mask, observations


def build_model(
    condition: str,
    *,
    weights: str,
    init_seed: int,
    model_dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
) -> tuple[OnlineTransientQueryClassifier, dict[str, Any]]:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    if condition == "panopticon_pretrained":
        weights_path = assert_safe_path(weights, purpose="official encoder weights")
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        backbone = _load_backbone(str(weights_path), strict=True)
        provenance = {
            "condition": condition,
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "full_backbone_train": True,
        }
    elif condition == "scratch":
        backbone = _load_backbone("random", strict=True)
        provenance = {
            "condition": condition,
            "weights_path": "",
            "weights_sha256": None,
            "full_backbone_train": True,
        }
    else:
        raise ValueError(f"Unknown condition: {condition}")
    model = OnlineTransientQueryClassifier(
        backbone,
        model_dim=model_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )
    provenance.update(
        {
            "init_seed": int(init_seed),
            "initial_state_sha256": state_dict_sha256(model.state_dict()),
            "head_parameter_signature": model_parameter_signature(model.head),
        }
    )
    return model, provenance


def build_loader(
    dataset: Query360Dataset,
    sampler: EpochOrderSampler,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": sampler,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": query360_collate,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs.update(
            prefetch_factor=int(prefetch_factor),
            persistent_workers=bool(persistent_workers),
        )
    return DataLoader(**kwargs)


def train_epoch(
    model: OnlineTransientQueryClassifier,
    loader: DataLoader,
    sampler: EpochOrderSampler,
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    epoch: int,
    seed: int,
    device: torch.device,
    amp_dtype: str,
    encoder_microbatch: int,
    auxiliary_weight: float,
    pos_weight: float,
    grad_clip: float,
    max_steps: int,
    log_interval: int,
) -> dict[str, Any]:
    sampler.set_epoch(epoch)
    random.seed(seed + epoch)
    np.random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + epoch)
    model.train()
    loss_sum = fused_sum = auxiliary_sum = 0.0
    rows_seen = observations_seen = steps = 0
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        if max_steps and step > max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        output, _, observations = model.forward_batch(
            batch,
            arm=arm,
            device=device,
            amp_dtype=amp_dtype,
            encoder_microbatch=encoder_microbatch,
        )
        labels = batch["labels"].to(
            device=device, dtype=output.fused_logits.dtype, non_blocking=True
        )
        loss = transient_query_loss(
            output,
            labels,
            auxiliary_weight=auxiliary_weight,
            pos_weight=pos_weight,
        )
        if not torch.isfinite(loss.total):
            raise RuntimeError(f"Non-finite online loss at epoch={epoch}, step={step}.")
        loss.total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        count = int(labels.numel())
        rows_seen += count
        observations_seen += observations
        steps += 1
        loss_sum += float(loss.total.detach()) * count
        fused_sum += float(loss.fused.detach()) * count
        auxiliary_sum += float(loss.auxiliary.detach()) * count
        if step % max(1, log_interval) == 0:
            print(
                f"[online-train] arm={arm} epoch={epoch} step={step} "
                f"rows={rows_seen} observations={observations_seen} "
                f"loss={loss_sum / rows_seen:.6f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if rows_seen == 0:
        raise RuntimeError("Online training processed no rows.")
    return {
        "loss": loss_sum / rows_seen,
        "fused_loss": fused_sum / rows_seen,
        "auxiliary_loss": auxiliary_sum / rows_seen,
        "rows": rows_seen,
        "observations": observations_seen,
        "steps": steps,
        "row_order_sha256": sampler.order_sha256(),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def evaluate(
    model: OnlineTransientQueryClassifier,
    loader: DataLoader,
    *,
    arm: str,
    device: torch.device,
    amp_dtype: str,
    encoder_microbatch: int,
    max_steps: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    valid_all: list[torch.Tensor] = []
    ids: list[str] = []
    plume_ids: list[str] = []
    cluster_ids: list[str] = []
    macro_region_ids: list[str] = []
    signatures: list[str] = []
    observations_seen = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            if max_steps and step > max_steps:
                break
            output, valid, observations = model.forward_batch(
                batch,
                arm=arm,
                device=device,
                amp_dtype=amp_dtype,
                encoder_microbatch=encoder_microbatch,
            )
            probabilities.append(torch.sigmoid(output.fused_logits).float().cpu())
            labels_all.append(batch["labels"].long().cpu())
            valid_all.append(valid.cpu())
            ids.extend(batch["ids"])
            plume_ids.extend(batch["plume_ids"])
            cluster_ids.extend(batch["cluster_ids"])
            macro_region_ids.extend(batch["macro_region_ids"])
            signatures.extend(batch["availability_signatures"])
            observations_seen += observations
    probability = torch.cat(probabilities).numpy()
    labels = torch.cat(labels_all)
    valid_mask = torch.cat(valid_all)
    metrics = stratified_binary_metrics(
        labels,
        probability,
        valid_mask,
        sensor_names=SENSOR_ORDER,
    )
    metrics["observations"] = int(observations_seen)
    predictions = pd.DataFrame(
        {
            "id": ids,
            "plume_id": plume_ids,
            "cluster_id": cluster_ids,
            "macro_region_id": macro_region_ids,
            "availability_signature": signatures,
            "label": labels.tolist(),
            "probability": probability,
            "prediction_at_0_5": (probability >= 0.5).astype(np.int64),
        }
    )
    return metrics, predictions


def run(args: argparse.Namespace) -> None:
    train_manifest = assert_safe_path(args.train_manifest, purpose="train manifest")
    val_manifest = assert_safe_path(
        args.inner_val_manifest, purpose="inner-validation manifest"
    )
    raw_cache_dir = assert_safe_path(args.raw_cache_dir, purpose="raw cache")
    output_dir = assert_safe_path(args.output_dir, purpose="online output")
    if not train_manifest.is_file() or not val_manifest.is_file():
        raise FileNotFoundError("Both train and inner-validation manifests are required.")
    if output_dir.joinpath("summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir}/summary.json exists; pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "condition": args.condition,
        },
    )

    cache = StrictHashedFileCache(raw_cache_dir)
    train_dataset = Query360Dataset(
        train_manifest,
        local_cache=cache,
        wv3_srf_csv=args.wv3_srf_csv,
        min_finite_fraction=args.min_finite_fraction,
    )
    val_dataset = Query360Dataset(
        val_manifest,
        local_cache=cache,
        wv3_srf_csv=args.wv3_srf_csv,
        min_finite_fraction=args.min_finite_fraction,
    )
    if set(train_dataset.frame["plume_id"]) & set(val_dataset.frame["plume_id"]):
        raise RuntimeError("Train/validation plume overlap detected.")
    device = configure_device(args.device)
    train_labels = train_dataset.frame["label"].astype(int).to_numpy()
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = negatives / max(positives, 1.0)
    condition_lr = (
        args.pretrained_backbone_lr
        if args.condition == "panopticon_pretrained"
        else args.scratch_backbone_lr
    )
    requested_arms = tuple(
        part.strip() for part in args.arms.split(",") if part.strip()
    )
    if not requested_arms or not set(requested_arms).issubset(ARMS):
        raise ValueError(f"--arms must be a subset of {ARMS}.")

    reference_initial_sha: Optional[str] = None
    arm_results = {}
    matched_orders: dict[str, list[str]] = {}
    try:
        for arm in requested_arms:
            model, provenance = build_model(
                args.condition,
                weights=args.weights,
                init_seed=args.init_seed,
                model_dim=args.model_dim,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout,
            )
            if reference_initial_sha is None:
                reference_initial_sha = provenance["initial_state_sha256"]
            elif provenance["initial_state_sha256"] != reference_initial_sha:
                raise RuntimeError("Online arms did not receive identical initialization.")
            model = model.to(device)
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": model.backbone.parameters(),
                        "lr": condition_lr,
                    },
                    {
                        "params": model.head.parameters(),
                        "lr": args.head_lr,
                    },
                ],
                weight_decay=args.weight_decay,
            )
            train_sampler = EpochOrderSampler(
                len(train_dataset), seed=args.batch_seed, shuffle=True
            )
            val_sampler = EpochOrderSampler(
                len(val_dataset), seed=0, shuffle=False
            )
            train_loader = build_loader(
                train_dataset,
                train_sampler,
                batch_size=args.row_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                pin_memory=device.type == "cuda",
            )
            val_loader = build_loader(
                val_dataset,
                val_sampler,
                batch_size=args.eval_row_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                pin_memory=device.type == "cuda",
            )
            history = []
            best: Optional[dict[str, Any]] = None
            arm_dir = output_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, args.epochs + 1):
                train_metrics = train_epoch(
                    model,
                    train_loader,
                    train_sampler,
                    optimizer,
                    arm=arm,
                    epoch=epoch,
                    seed=args.batch_seed,
                    device=device,
                    amp_dtype=args.amp_dtype,
                    encoder_microbatch=args.encoder_microbatch,
                    auxiliary_weight=args.auxiliary_weight,
                    pos_weight=pos_weight,
                    grad_clip=args.grad_clip,
                    max_steps=args.max_train_steps,
                    log_interval=args.log_interval,
                )
                validation, predictions = evaluate(
                    model,
                    val_loader,
                    arm=arm,
                    device=device,
                    amp_dtype=args.amp_dtype,
                    encoder_microbatch=args.encoder_microbatch,
                    max_steps=args.max_eval_steps,
                )
                record = {
                    "epoch": int(epoch),
                    "train": train_metrics,
                    "validation": validation,
                }
                history.append(record)
                ap = validation["overall"]["ap"]
                if ap is None:
                    raise RuntimeError("Online validation AP is undefined.")
                if best is None or float(ap) > float(
                    best["validation"]["overall"]["ap"]
                ):
                    best = copy.deepcopy(record)
                    checkpoint = {
                        "schema_version": "query360-online-checkpoint-v1",
                        "script_version": SCRIPT_VERSION,
                        "condition": args.condition,
                        "arm": arm,
                        "epoch": epoch,
                        "model": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "initialization": provenance,
                        "validation": validation,
                    }
                    atomic_torch_save(
                        arm_dir / "checkpoint_best_ap.pt", checkpoint
                    )
                    predictions["arm"] = arm
                    predictions["condition"] = args.condition
                    predictions["epoch"] = epoch
                    atomic_csv_write(
                        arm_dir / "validation_best_ap_predictions.csv",
                        predictions,
                    )
                atomic_json_write(arm_dir / "metrics_history.json", history)
                overall = validation["overall"]
                print(
                    f"[online] condition={args.condition} arm={arm} "
                    f"epoch={epoch}/{args.epochs} "
                    f"AP={overall['ap']:.6f} AUC={overall['auc']:.6f} "
                    f"F1@.5={overall['macro_f1_at_0_5']:.6f} "
                    f"bestF1={overall['best_macro_f1']:.6f}",
                    flush=True,
                )
            if best is None:
                raise RuntimeError(f"{arm}: no online epoch result.")
            arm_results[arm] = best
            matched_orders[arm] = [
                record["train"]["row_order_sha256"] for record in history
            ]
            del model, optimizer, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if len(set(tuple(values) for values in matched_orders.values())) != 1:
            raise RuntimeError("Online arms saw different epoch row orders.")
        summary = {
            "schema_version": "query360-online-summary-v1",
            "script_version": SCRIPT_VERSION,
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "condition": args.condition,
            "full_backbone_train": True,
            "initial_state_sha256": reference_initial_sha,
            "arms": arm_results,
            "matched_epoch_row_order": next(iter(matched_orders.values())),
            "train_manifest": {
                "path": str(train_manifest),
                "sha256": sha256_file(train_manifest),
                "rows": len(train_dataset),
            },
            "inner_val_manifest": {
                "path": str(val_manifest),
                "sha256": sha256_file(val_manifest),
                "rows": len(val_dataset),
            },
            "training_contract": {
                "requested_arms": list(requested_arms),
                "epochs": args.epochs,
                "max_train_steps": args.max_train_steps,
                "max_eval_steps": args.max_eval_steps,
                "row_batch_size": args.row_batch_size,
                "eval_row_batch_size": args.eval_row_batch_size,
                "encoder_microbatch": args.encoder_microbatch,
                "batch_seed": args.batch_seed,
                "backbone_lr": condition_lr,
                "head_lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "auxiliary_weight": args.auxiliary_weight,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
                "amp_dtype": args.amp_dtype,
            },
            "safety": {
                "external_test_manifest_read": False,
                "checkpoint_selected_on_inner_validation_ap": True,
                "raw_cache_only_after_warmup": True,
            },
        }
        atomic_json_write(output_dir / "summary.json", summary)
        atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            },
        )
    except Exception as error:
        atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--inner-val-manifest", required=True)
    parser.add_argument("--raw-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--condition",
        choices=["panopticon_pretrained", "scratch"],
        required=True,
    )
    parser.add_argument("--arms", default="current_only,transient_query")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--wv3-srf-csv", default=str(DEFAULT_WV3_SRF))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--row-batch-size", type=int, default=24)
    parser.add_argument("--eval-row-batch-size", type=int, default=32)
    parser.add_argument("--encoder-microbatch", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pretrained-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--scratch-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--auxiliary-weight", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-finite-fraction", type=float, default=0.05)
    parser.add_argument("--init-seed", type=int, default=360)
    parser.add_argument("--batch-seed", type=int, default=17)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
