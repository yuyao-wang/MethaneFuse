#!/usr/bin/env python3
"""Frozen-backbone, final-patch RCTP transfer diagnostic on real L89.

This module answers one deliberately narrow question after a global-CLS RCTP
transfer failure:

    Did the continued-pretraining checkpoint retain useful *local* temporal
    evidence that was destroyed by CLS pooling?

It has two stages.

``extract``
    Reuses an audited six-visit L89 CLS cache for row identity, validity masks,
    preprocessing metadata, and a frozen role-only CLS checkpoint.  It runs the
    declared Panopticon weights once more, reads only the final normalized patch
    tokens, and forms same-location temporal evidence

        delta[p] = token(t0, p) - mean_valid_history token(t, p).

    A fixed energy top-k is pooled and projected by a shared seeded orthogonal
    projection.  Only the resulting local evidence is persisted; raw images and
    the large patch-token tensor are not.  Static CLS features are never an
    input to this local evidence path.

``train-compare``
    Trains a zero-initialized, bias-free linear residual on top of each arm's
    already-frozen role-only CLS logit.  Therefore epoch zero is bit-exact to
    that arm's base logit.  P0/P4/P5 local heads have byte-identical initial
    states, batches, optimizer settings, and parameter shapes.  Both row and
    canonical-event-balanced metrics are reported.

This is train/development-only engineering evidence.  Test/sealed paths are
rejected and no result from this script is an unbiased final estimate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


_script_path = Path(__file__).resolve()
REPO_ROOT = Path(
    os.environ.get(
        "REPO_ROOT",
        str(_script_path.parents[2] if len(_script_path.parents) > 2 else Path.cwd()),
    )
).expanduser().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as l89,
)
from research.pretraining_20260727.legacy360_patch_cache import (  # noqa: E402
    deterministic_orthogonal_projection,
    projection_metadata,
    tensor_sha256,
)
from Upgraded_dataset.dino_classifier_head_l89_temporal_satmae import (  # noqa: E402
    load_backbone,
)


SCRIPT_VERSION = "l89-rctp-final-patch-local-followup-v1"
CACHE_VERSION = "l89-rctp-final-patch-evidence-cache-v1"
HEAD_VERSION = "l89-rctp-local-logit-residual-v1"
FORBIDDEN_PATH_TOKENS = ("test", "sealed", "holdout")
DEFAULT_CACHE_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
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
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def assert_development_path(path: Path, *, purpose: str) -> None:
    lowered = str(path).casefold()
    hits = [token for token in FORBIDDEN_PATH_TOKENS if token in lowered]
    if hits:
        raise ValueError(
            f"{purpose} path contains held-out token(s) {hits}: {path}"
        )


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=device.type == "cuda")


def compute_local_patch_evidence(
    patch_tokens: torch.Tensor,
    unique_mask: torch.Tensor,
    projection: torch.Tensor,
    *,
    t0_index: int,
    topk_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pool local ``t0 - matched-history`` evidence without any CLS input.

    Parameters
    ----------
    patch_tokens
        Final normalized patch tokens with shape ``[B,T,P,D]``.
    unique_mask
        Valid non-duplicate observation mask ``[B,T]``.
    projection
        Fixed orthogonal projection ``[D,d]`` shared by all experimental arms.

    Returns
    -------
    evidence
        ``[B,3*d]`` containing projected top-k signed delta, projected top-k
        absolute delta, and projected global signed delta.
    history_count
        Number of valid unique history observations for every row.
    keep
        Fixed number of selected patches.
    """

    if patch_tokens.ndim != 4:
        raise ValueError("patch_tokens must have shape [B,T,P,D]")
    rows, roles, patches, feature_dim = patch_tokens.shape
    if unique_mask.shape != (rows, roles) or unique_mask.dtype != torch.bool:
        raise ValueError("unique_mask must be boolean [B,T]")
    if not 0 <= int(t0_index) < roles:
        raise ValueError("t0_index is out of range")
    if not unique_mask[:, int(t0_index)].all():
        raise ValueError("every local-evidence row requires a valid unique t0")
    if projection.ndim != 2 or tuple(projection.shape)[0] != feature_dim:
        raise ValueError("projection shape does not match patch feature dim")
    if not 0.0 < float(topk_fraction) <= 1.0:
        raise ValueError("topk_fraction must be in (0,1]")

    history_indices = [
        index for index in range(roles) if index != int(t0_index)
    ]
    if not history_indices:
        raise ValueError("at least one history role is required")
    history_valid = unique_mask[:, history_indices]
    history_count = history_valid.sum(dim=1)
    history_weight = history_valid.to(patch_tokens.dtype)[:, :, None, None]
    history = patch_tokens[:, history_indices]
    history_mean = (history * history_weight).sum(dim=1) / history_weight.sum(
        dim=1
    ).clamp_min(1.0)
    delta = patch_tokens[:, int(t0_index)] - history_mean
    has_history = history_count.gt(0)
    delta = torch.where(
        has_history[:, None, None],
        delta,
        torch.zeros_like(delta),
    )

    # Ranking in the original final-token space avoids allowing a random
    # projection to choose the spatial support.  The projection is applied
    # only after deterministic pooling.
    energy = delta.float().square().mean(dim=-1)
    keep = max(1, int(math.ceil(patches * float(topk_fraction))))
    selected = torch.topk(energy, k=keep, dim=1, largest=True).indices
    gather = selected[:, :, None].expand(rows, keep, feature_dim)
    selected_delta = delta.gather(1, gather).float()
    top_signed = selected_delta.mean(dim=1)
    top_absolute = selected_delta.abs().mean(dim=1)
    global_signed = delta.float().mean(dim=1)

    projection = projection.to(
        device=patch_tokens.device, dtype=torch.float32
    )
    evidence = torch.cat(
        (
            top_signed @ projection,
            top_absolute @ projection,
            global_signed @ projection,
        ),
        dim=-1,
    )
    evidence = torch.where(
        has_history[:, None], evidence, torch.zeros_like(evidence)
    )
    return evidence, history_count, keep


class LocalLogitResidual(nn.Module):
    """Bias-free local-only residual; construction is an exact no-op."""

    def __init__(self, evidence_dim: int, *, residual_cap: float = 1.5):
        super().__init__()
        if evidence_dim < 1:
            raise ValueError("evidence_dim must be positive")
        if residual_cap <= 0:
            raise ValueError("residual_cap must be positive")
        self.residual_cap = float(residual_cap)
        self.readout = nn.Linear(int(evidence_dim), 1, bias=False)
        nn.init.zeros_(self.readout.weight)

    @property
    def exact_noop(self) -> bool:
        return bool(torch.count_nonzero(self.readout.weight).item() == 0)

    def forward(
        self, base_logits: torch.Tensor, local_evidence: torch.Tensor
    ) -> torch.Tensor:
        if base_logits.ndim != 1:
            raise ValueError("base_logits must have shape [B]")
        if (
            local_evidence.ndim != 2
            or local_evidence.shape[0] != base_logits.shape[0]
        ):
            raise ValueError("local_evidence must have shape [B,D]")
        raw = self.readout(local_evidence).squeeze(-1)
        residual = self.residual_cap * torch.tanh(raw / self.residual_cap)
        return base_logits + residual


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    # Historical l89 head checkpoints accidentally serialized the argparse
    # ``handler`` function as ``__main__.train_heads``.  Supplying that exact
    # symbol is a read-only compatibility shim; the object is never called.
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "train_heads"):
        setattr(main_module, "train_heads", l89.train_heads)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a mapping")
    return payload


def _build_frozen_base_head(
    checkpoint: Mapping[str, Any],
    cls_cache: Mapping[str, Any],
) -> l89.RaggedCurrentQueryHead:
    arguments = checkpoint.get("args")
    state = checkpoint.get("model")
    if not isinstance(arguments, Mapping) or not isinstance(state, Mapping):
        raise ValueError("base head checkpoint lacks args/model")
    if str(checkpoint.get("arm")) != "role_only":
        raise ValueError("base head checkpoint must be the role_only arm")
    model = l89.RaggedCurrentQueryHead(
        feature_dim=int(cls_cache["features"].shape[-1]),
        num_roles=int(cls_cache["features"].shape[1]),
        model_dim=int(arguments["model_dim"]),
        num_heads=int(arguments["num_heads"]),
        depth=2,
        mlp_ratio=float(arguments["mlp_ratio"]),
        dropout=float(arguments["dropout"]),
        periods_days=tuple(
            float(value)
            for value in str(arguments["delta_periods"]).split(",")
        ),
        t0_index=int(cls_cache["t0_index"]),
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def frozen_base_logits(
    checkpoint: Mapping[str, Any],
    cls_cache: Mapping[str, Any],
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    """Reproduce the selected role-only base head on CPU."""

    model = _build_frozen_base_head(checkpoint, cls_cache)
    usable = l89.select_usable_rows(cls_cache)
    if int(usable.numel()) != int(cls_cache["features"].shape[0]):
        raise ValueError(
            "patch-local v1 requires every cached row to have a valid unique t0"
        )
    data = l89.take_rows(cls_cache, usable)
    role_index = cls_cache["role_index"].long()
    t0_index = int(cls_cache["t0_index"])
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in l89.fixed_epoch_batches(
            len(data["labels"]),
            batch_size=int(batch_size),
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            features, valid, delta_days, enable_delta = l89.prepare_arm_inputs(
                data["features"][indices],
                data["valid_mask"][indices],
                data["unique_mask"][indices],
                data["delta_days"][indices],
                arm="role_only",
                t0_index=t0_index,
            )
            outputs.append(
                model(
                    features,
                    valid,
                    role_index,
                    delta_days,
                    enable_delta=enable_delta,
                ).float()
            )
    return torch.cat(outputs, dim=0)


def _validate_all_local_paths(
    frame: pd.DataFrame,
    path_columns: Sequence[str],
    *,
    allowed_root: Path,
) -> dict[str, Any]:
    allowed_root = allowed_root.expanduser().resolve()
    paths: set[Path] = set()
    for column in path_columns:
        for value in frame[column].fillna("").astype(str):
            if not value.strip():
                continue
            path = Path(os.path.abspath(os.path.expanduser(value.strip())))
            try:
                path.relative_to(allowed_root)
            except ValueError as exc:
                raise ValueError(
                    f"{column} escapes required local root {allowed_root}: {path}"
                ) from exc
            paths.add(path)
    missing = [str(path) for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} staged inputs are missing; examples={missing[:5]}"
        )
    logical_bytes = sum(path.stat().st_size for path in paths)
    return {
        "allowed_root": str(allowed_root),
        "unique_files": len(paths),
        "logical_bytes": int(logical_bytes),
        "all_inputs_local": True,
    }


def _verify_base_validation_predictions(
    base_logits: torch.Tensor,
    cls_cache: Mapping[str, Any],
    predictions_path: Optional[Path],
) -> Optional[dict[str, Any]]:
    if predictions_path is None:
        return None
    frame = pd.read_csv(predictions_path)
    if len(frame) != len(base_logits):
        raise ValueError("base validation prediction row count differs")
    if frame["id"].astype(str).tolist() != [
        str(value) for value in cls_cache["ids"]
    ]:
        raise ValueError("base validation prediction IDs differ")
    expected = frame["probability"].to_numpy(dtype=np.float64)
    observed = torch.sigmoid(base_logits).numpy().astype(np.float64)
    maximum = float(np.max(np.abs(expected - observed)))
    if maximum > 2.0e-6:
        raise ValueError(
            f"reproduced base probabilities differ by {maximum:.3e}"
        )
    return {
        "path": str(predictions_path),
        "sha256": sha256_file(predictions_path),
        "max_abs_probability_error": maximum,
        "verified": True,
    }


def command_extract(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    csv_path = Path(args.csv).expanduser().resolve()
    cls_cache_path = Path(args.cls_cache).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    base_head_path = Path(args.base_head_checkpoint).expanduser().resolve()
    output_path = Path(args.output_cache).expanduser().resolve()
    for path, purpose in (
        (csv_path, "CSV"),
        (cls_cache_path, "CLS cache"),
        (base_head_path, "base head"),
        (output_path, "output cache"),
    ):
        assert_development_path(path, purpose=purpose)
    for path in (csv_path, cls_cache_path, weights_path, base_head_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    cls_cache = _load_torch_payload(cls_cache_path)
    l89.validate_cache_payload(
        cls_cache, path=cls_cache_path, expected_split=args.split
    )
    if str(cls_cache["csv_path"]) != str(csv_path):
        raise ValueError("CLS cache CSV path differs from --csv")
    if str(cls_cache["weights_path"]) != str(weights_path):
        raise ValueError("CLS cache weights path differs from --weights")
    if sha256_file(csv_path) != str(cls_cache["csv_sha256"]):
        raise ValueError("CSV SHA differs from audited CLS cache")
    if sha256_file(weights_path) != str(cls_cache["weights_sha256"]):
        raise ValueError("weights SHA differs from audited CLS cache")

    contract = cls_cache["input_contract"]
    frame = pd.read_csv(csv_path, low_memory=False)
    if len(frame) != int(cls_cache["features"].shape[0]):
        raise ValueError("CSV and CLS cache row counts differ")
    ids = l89.string_column(
        frame, "id", fallback=l89.string_column(frame, "plume_id")
    )
    if [str(value) for value in ids] != [
        str(value) for value in cls_cache["ids"]
    ]:
        raise ValueError("CSV and CLS cache row identities differ")

    path_columns = tuple(str(value) for value in contract["path_columns"])
    local_audit = _validate_all_local_paths(
        frame,
        path_columns,
        allowed_root=Path(args.required_local_root),
    )
    dataset = l89.L89FrameCacheDataset(
        csv_path,
        frame,
        path_columns=path_columns,
        band_indices=tuple(int(value) for value in contract["band_indices"]),
        mean=tuple(float(value) for value in contract["normalization_mean"]),
        std=tuple(float(value) for value in contract["normalization_std"]),
        image_size=int(contract["image_size"]),
        min_valid_fraction=float(contract["min_valid_fraction"]),
        validity_band_index=int(contract["validity_band_index"]),
        local_file_cache=None,
        local_cache_bypass_root=Path(args.required_local_root),
        zero_invalid_pixels=bool(contract["zero_invalid_pixels"]),
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = DataLoader(dataset, **loader_kwargs)

    base_checkpoint = _load_torch_payload(base_head_path)
    base_logits = frozen_base_logits(
        base_checkpoint, cls_cache, batch_size=args.base_eval_batch_size
    )
    prediction_audit = _verify_base_validation_predictions(
        base_logits,
        cls_cache,
        (
            Path(args.base_validation_predictions).expanduser().resolve()
            if args.base_validation_predictions
            else None
        ),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        required_free = float(args.min_cuda_free_gib) * (2**30)
        if free_bytes < required_free:
            raise RuntimeError(
                f"{device} has only {free_bytes / 2**30:.2f} GiB free; "
                f"protocol requires >= {args.min_cuda_free_gib:.2f} GiB "
                "before loading the frozen backbone"
            )
    backbone = load_backbone(
        str(weights_path), device=device, debug=args.debug
    ).to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise AssertionError("patch extraction backbone must be fully frozen")
    projection = deterministic_orthogonal_projection(
        int(backbone.embed_dim),
        int(args.projection_dim),
        seed=int(args.projection_seed),
    )
    projection_info = projection_metadata(
        projection, seed=int(args.projection_seed)
    )

    rows = len(frame)
    evidence_dim = int(args.projection_dim) * 3
    evidence = torch.zeros(rows, evidence_dim, dtype=torch.float16)
    history_count = torch.zeros(rows, dtype=torch.int8)
    seen = torch.zeros(rows, dtype=torch.bool)
    expected_unique = cls_cache["unique_mask"].bool()
    channel_ids = dataset.channel_ids
    patch_count: Optional[int] = None
    keep_count: Optional[int] = None
    started = time.monotonic()
    maximum_allocated = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            indices, images, image_valid, _fractions, _status = batch
            indices = indices.long()
            if seen[indices].any():
                raise RuntimeError("a patch row was emitted twice")
            # A changed reader/preprocessing path must fail before feature use.
            if not torch.equal(
                image_valid.bool(), cls_cache["image_valid_mask"][indices].bool()
            ):
                raise ValueError("online image validity differs from CLS cache")
            batch_rows, roles, channels, height, width = images.shape
            flat_images = images.reshape(
                batch_rows * roles, channels, height, width
            ).to(device, non_blocking=True)
            flat_ids = (
                channel_ids.view(1, -1)
                .expand(batch_rows * roles, -1)
                .clone()
                .to(device, non_blocking=True)
            )
            with autocast_context(device, args.amp_dtype):
                output = backbone.forward_features(
                    {"imgs": flat_images, "chn_ids": flat_ids}
                )
                patches = output["x_norm_patchtokens"].reshape(
                    batch_rows, roles, -1, int(backbone.embed_dim)
                )
            current_patch_count = int(patches.shape[2])
            if patch_count is None:
                patch_count = current_patch_count
            elif patch_count != current_patch_count:
                raise ValueError("final patch-token count changed between batches")
            batch_evidence, batch_history, current_keep = (
                compute_local_patch_evidence(
                    patches,
                    move_tensor(expected_unique[indices], device),
                    move_tensor(projection, device),
                    t0_index=int(cls_cache["t0_index"]),
                    topk_fraction=float(args.topk_fraction),
                )
            )
            if keep_count is None:
                keep_count = current_keep
            elif keep_count != current_keep:
                raise AssertionError("fixed top-k count changed")
            evidence[indices] = batch_evidence.cpu().to(torch.float16)
            history_count[indices] = batch_history.cpu().to(torch.int8)
            seen[indices] = True
            if device.type == "cuda":
                maximum_allocated = max(
                    maximum_allocated, int(torch.cuda.max_memory_allocated(device))
                )
                if maximum_allocated > float(args.max_cuda_allocated_gib) * (
                    2**30
                ):
                    raise RuntimeError(
                        "measured CUDA allocation exceeded the protocol cap: "
                        f"{maximum_allocated / 2**30:.2f} GiB > "
                        f"{args.max_cuda_allocated_gib:.2f} GiB"
                    )
            if (
                batch_index % max(1, int(args.log_interval)) == 0
                or batch_index == len(loader)
            ):
                print(
                    f"[final-patch] arm={args.arm} split={args.split} "
                    f"batch={batch_index}/{len(loader)} "
                    f"rows={int(seen.sum())}/{rows} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if not seen.all():
        missing = torch.nonzero(~seen, as_tuple=False).flatten().tolist()[:20]
        raise RuntimeError(f"patch evidence missed rows {missing}")
    no_history = int(history_count.eq(0).sum())

    identity = {
        "ids": [str(value) for value in cls_cache["ids"]],
        "plume_ids": [str(value) for value in cls_cache["plume_ids"]],
        "event_ids": [str(value) for value in cls_cache["event_ids"]],
        "labels_sha256": tensor_sha256(cls_cache["labels"].long()),
        "unique_mask_sha256": tensor_sha256(expected_unique),
    }
    identity_sha = canonical_json_sha256(identity)
    payload = {
        "format_version": CACHE_VERSION,
        "script_version": SCRIPT_VERSION,
        "arm": str(args.arm),
        "split": str(args.split),
        "local_evidence": evidence,
        "base_logits": base_logits.float(),
        "labels": cls_cache["labels"].long(),
        "ids": identity["ids"],
        "plume_ids": identity["plume_ids"],
        "event_ids": identity["event_ids"],
        "history_count": history_count,
        "identity_sha256": identity_sha,
        "evidence_sha256": tensor_sha256(evidence),
        "base_logits_sha256": tensor_sha256(base_logits.float()),
        "projection": projection_info,
        "local_evidence_contract": {
            "backbone_trainable_parameters": 0,
            "backbone_mode": "eval_inference_only",
            "token_layer": "x_norm_patchtokens_final",
            "static_cls_is_local_head_input": False,
            "delta": "t0_patch-minus-masked-mean-valid-unique-history_patch",
            "spatial_correspondence": "same_patch_index_within_aligned_224_crop",
            "topk_rank_space": "original_768d_final_delta_rms",
            "topk_fraction": float(args.topk_fraction),
            "patch_count": int(patch_count or 0),
            "topk_count": int(keep_count or 0),
            "pooled_components": [
                "topk_signed_delta",
                "topk_absolute_delta",
                "global_signed_delta",
            ],
            "projection_dim_each_component": int(args.projection_dim),
            "evidence_dim": evidence_dim,
            "zero_evidence_when_no_history": True,
        },
        "provenance": {
            "csv": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "weights": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "cls_cache": str(cls_cache_path),
            "cls_cache_sha256": sha256_file(cls_cache_path),
            "base_head_checkpoint": str(base_head_path),
            "base_head_checkpoint_sha256": sha256_file(base_head_path),
            "base_head_arm": base_checkpoint.get("arm"),
            "base_head_epoch": base_checkpoint.get("epoch"),
            "base_validation_prediction_audit": prediction_audit,
            "local_input_audit": local_audit,
        },
        "rows": rows,
        "no_history_rows": no_history,
        "elapsed_seconds": float(time.monotonic() - started),
        "cuda_max_memory_allocated_bytes": int(maximum_allocated),
        "test_or_sealed_read": False,
    }
    atomic_torch(output_path, payload)
    summary = {
        "status": "complete",
        "cache": str(output_path),
        "cache_sha256": sha256_file(output_path),
        "arm": str(args.arm),
        "split": str(args.split),
        "rows": rows,
        "evidence_shape": list(evidence.shape),
        "evidence_sha256": payload["evidence_sha256"],
        "base_logits_sha256": payload["base_logits_sha256"],
        "identity_sha256": identity_sha,
        "no_history_rows": no_history,
        "local_evidence_contract": payload["local_evidence_contract"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "cuda_max_memory_allocated_bytes": maximum_allocated,
        "test_or_sealed_read": False,
    }
    atomic_json(output_path.with_suffix(output_path.suffix + ".json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def validate_local_cache(
    payload: Mapping[str, Any], path: Path, *, split: str
) -> None:
    required = {
        "format_version",
        "arm",
        "split",
        "local_evidence",
        "base_logits",
        "labels",
        "ids",
        "plume_ids",
        "event_ids",
        "history_count",
        "identity_sha256",
        "evidence_sha256",
        "base_logits_sha256",
        "projection",
        "local_evidence_contract",
        "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if payload["format_version"] != CACHE_VERSION:
        raise ValueError(f"{path}: unsupported cache version")
    if payload["split"] != split:
        raise ValueError(f"{path}: expected split={split}")
    evidence = payload["local_evidence"]
    rows = int(evidence.shape[0])
    if evidence.ndim != 2 or evidence.dtype != torch.float16:
        raise ValueError(f"{path}: invalid local evidence")
    if payload["base_logits"].shape != (rows,):
        raise ValueError(f"{path}: invalid base logits")
    if payload["labels"].shape != (rows,):
        raise ValueError(f"{path}: invalid labels")
    if any(len(payload[key]) != rows for key in ("ids", "plume_ids", "event_ids")):
        raise ValueError(f"{path}: identity lengths differ")
    if tensor_sha256(evidence) != payload["evidence_sha256"]:
        raise ValueError(f"{path}: evidence SHA mismatch")
    if tensor_sha256(payload["base_logits"].float()) != payload["base_logits_sha256"]:
        raise ValueError(f"{path}: base-logit SHA mismatch")
    contract = payload["local_evidence_contract"]
    if contract.get("token_layer") != "x_norm_patchtokens_final":
        raise ValueError(f"{path}: follow-up must use final patch tokens")
    if contract.get("static_cls_is_local_head_input") is not False:
        raise ValueError(f"{path}: static CLS leaked into local head input")
    if int(contract.get("backbone_trainable_parameters", -1)) != 0:
        raise ValueError(f"{path}: backbone was not fully frozen")


def event_balanced_training_weights(
    labels: torch.Tensor, event_ids: Sequence[str]
) -> torch.Tensor:
    event = torch.from_numpy(l89.event_balanced_row_weights(event_ids)).float()
    labels = labels.long()
    positive_mass = event[labels.eq(1)].sum()
    negative_mass = event[labels.eq(0)].sum()
    if positive_mass <= 0 or negative_mass <= 0:
        raise ValueError("both classes need positive event mass")
    class_factor = torch.ones_like(event)
    class_factor[labels.eq(1)] = negative_mass / positive_mass
    weights = event * class_factor
    return weights / weights.mean().clamp_min(1e-12)


def metrics_from_logits(
    labels: torch.Tensor,
    logits: torch.Tensor,
    event_ids: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    target = labels.detach().cpu().numpy().astype(np.int64)
    probability = torch.sigmoid(logits.float()).detach().cpu().numpy()
    event_weights = l89.event_balanced_row_weights(event_ids)
    row_threshold, _ = l89.best_weighted_positive_f1_threshold(
        target, probability, np.ones_like(probability, dtype=np.float64)
    )
    event_threshold, _ = l89.best_weighted_positive_f1_threshold(
        target, probability, event_weights
    )

    def at_threshold(threshold: float, weights: Optional[np.ndarray]):
        prediction = (probability >= float(threshold)).astype(np.int64)
        kwargs = (
            {"sample_weight": weights} if weights is not None else {}
        )
        return {
            "threshold": float(threshold),
            "positive_f1": float(
                f1_score(
                    target,
                    prediction,
                    average="binary",
                    zero_division=0,
                    **kwargs,
                )
            ),
            "macro_f1": float(
                f1_score(
                    target,
                    prediction,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                    **kwargs,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    target, prediction, sample_weight=weights
                )
            ),
            "predicted_positive_rate": float(prediction.mean()),
        }

    metrics = {
        "rows": int(len(target)),
        "events": int(len(set(str(value) for value in event_ids))),
        "row_ap": float(average_precision_score(target, probability)),
        "row_auc": float(roc_auc_score(target, probability)),
        "row_at_0p5": at_threshold(0.5, None),
        "row_at_row_selected": at_threshold(row_threshold, None),
        "row_at_event_selected": at_threshold(event_threshold, None),
        "event_balanced_ap": float(
            average_precision_score(
                target, probability, sample_weight=event_weights
            )
        ),
        "event_balanced_auc": float(
            roc_auc_score(target, probability, sample_weight=event_weights)
        ),
        "event_balanced_at_0p5": at_threshold(0.5, event_weights),
        "event_balanced_at_event_selected": at_threshold(
            event_threshold, event_weights
        ),
    }
    return metrics, probability


def _cache_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ids": [str(value) for value in payload["ids"]],
        "plume_ids": [str(value) for value in payload["plume_ids"]],
        "event_ids": [str(value) for value in payload["event_ids"]],
        "labels": payload["labels"].long().tolist(),
    }


def train_one_arm(
    name: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    initial_state: Mapping[str, torch.Tensor],
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    evidence_dim = int(train_cache["local_evidence"].shape[1])
    model = LocalLogitResidual(
        evidence_dim, residual_cap=float(args.residual_cap)
    ).to(device)
    model.load_state_dict(initial_state, strict=True)
    if not model.exact_noop:
        raise AssertionError("local residual initial state must be exact zero")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    labels = train_cache["labels"].long()
    sample_weights = event_balanced_training_weights(
        labels, train_cache["event_ids"]
    )
    base_train = train_cache["base_logits"].float()
    evidence_train = train_cache["local_evidence"].float()
    base_val = val_cache["base_logits"].float().to(device)
    evidence_val = val_cache["local_evidence"].float().to(device)
    val_labels = val_cache["labels"].long()

    with torch.inference_mode():
        epoch_zero_logits = model(base_val, evidence_val)
    if not torch.equal(epoch_zero_logits.cpu(), val_cache["base_logits"].float()):
        raise AssertionError("epoch zero is not bit-exact to base logits")
    epoch_zero_metrics, epoch_zero_probability = metrics_from_logits(
        val_labels, epoch_zero_logits.cpu(), val_cache["event_ids"]
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "optimizer_steps": 0,
            "validation": epoch_zero_metrics,
            "exact_base_logit": True,
        }
    ]
    best = copy.deepcopy(history[0])
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_probability = epoch_zero_probability
    started = time.monotonic()

    for epoch in range(1, int(args.epochs) + 1):
        set_seed(int(args.seed))
        model.train()
        batches = l89.fixed_epoch_batches(
            len(labels),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            epoch=epoch,
            shuffle=True,
        )
        total_loss = 0.0
        rows_seen = 0
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            batch_base = base_train[indices].to(device)
            batch_evidence = evidence_train[indices].to(device)
            batch_labels = labels[indices].float().to(device)
            batch_weights = sample_weights[indices].to(device)
            logits = model(batch_base, batch_evidence)
            loss_per_row = F.binary_cross_entropy_with_logits(
                logits, batch_labels, reduction="none"
            )
            loss = (loss_per_row * batch_weights).sum() / batch_weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss")
            loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.grad_clip)
                )
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            rows_seen += len(indices)
        model.eval()
        with torch.inference_mode():
            val_logits = model(base_val, evidence_val)
        metrics, probability = metrics_from_logits(
            val_labels, val_logits.cpu(), val_cache["event_ids"]
        )
        record = {
            "epoch": epoch,
            "optimizer_steps": len(batches),
            "train_rows_seen": rows_seen,
            "event_class_balanced_train_loss": total_loss / rows_seen,
            "validation": metrics,
            "elapsed_seconds": float(time.monotonic() - started),
            "exact_base_logit": False,
        }
        history.append(record)
        if (
            metrics["event_balanced_ap"]
            > best["validation"]["event_balanced_ap"]
        ):
            best = copy.deepcopy(record)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_probability = probability
        print(
            f"[local-head] arm={name} epoch={epoch}/{args.epochs} "
            f"event_AP={metrics['event_balanced_ap']:.6f} "
            f"event_macroF1="
            f"{metrics['event_balanced_at_event_selected']['macro_f1']:.6f} "
            f"row_AP={metrics['row_ap']:.6f}",
            flush=True,
        )

    arm_dir = output_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": HEAD_VERSION,
        "arm": name,
        "epoch": int(best["epoch"]),
        "model": best_state,
        "initial_state_sha256": l89.state_dict_sha256(initial_state),
        "selection_metric": "event_balanced_ap",
        "validation": best["validation"],
        "train_cache_sha256": train_cache["_file_sha256"],
        "val_cache_sha256": val_cache["_file_sha256"],
        "static_cls_is_local_head_input": False,
        "base_logits_frozen": True,
        "backbone_frozen": True,
    }
    atomic_torch(arm_dir / "checkpoint_best_event_ap.pt", checkpoint)
    atomic_json(arm_dir / "metrics_history.json", history)
    prediction_frame = pd.DataFrame(
        {
            "id": val_cache["ids"],
            "plume_id": val_cache["plume_ids"],
            "event_id": val_cache["event_ids"],
            "label": val_labels.tolist(),
            "base_probability": torch.sigmoid(
                val_cache["base_logits"].float()
            ).numpy(),
            "local_probability": best_probability,
            "best_epoch": int(best["epoch"]),
            "arm": name,
        }
    )
    l89.atomic_csv_write(
        arm_dir / "validation_best_event_ap_predictions.csv",
        prediction_frame,
    )
    return {
        "arm": name,
        "epoch_zero": history[0],
        "best": best,
        "event_balanced_ap_delta": float(
            best["validation"]["event_balanced_ap"]
            - history[0]["validation"]["event_balanced_ap"]
        ),
        "event_balanced_macro_f1_delta": float(
            best["validation"]["event_balanced_at_event_selected"]["macro_f1"]
            - history[0]["validation"]["event_balanced_at_event_selected"][
                "macro_f1"
            ]
        ),
        "row_ap_delta": float(
            best["validation"]["row_ap"]
            - history[0]["validation"]["row_ap"]
        ),
    }


def command_train_compare(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    assert_development_path(output_dir, purpose="output")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arms: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for name, train_value, val_value in args.arm:
        train_path = Path(train_value).expanduser().resolve()
        val_path = Path(val_value).expanduser().resolve()
        assert_development_path(train_path, purpose=f"{name} train cache")
        assert_development_path(val_path, purpose=f"{name} val cache")
        train = dict(_load_torch_payload(train_path))
        val = dict(_load_torch_payload(val_path))
        validate_local_cache(train, train_path, split="train")
        validate_local_cache(val, val_path, split="val")
        if train["arm"] != name or val["arm"] != name:
            raise ValueError(f"{name}: cache arm labels differ")
        for key in ("projection", "local_evidence_contract"):
            if train[key] != val[key]:
                raise ValueError(f"{name}: train/val differ in {key}")
        train["_file_sha256"] = sha256_file(train_path)
        val["_file_sha256"] = sha256_file(val_path)
        arms.append((str(name), train, val))
    if len({name for name, _, _ in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    reference_train = _cache_identity(arms[0][1])
    reference_val = _cache_identity(arms[0][2])
    reference_projection = arms[0][1]["projection"]
    for name, train, val in arms[1:]:
        if _cache_identity(train) != reference_train:
            raise ValueError(f"{name}: train row identities differ")
        if _cache_identity(val) != reference_val:
            raise ValueError(f"{name}: val row identities differ")
        if train["projection"] != reference_projection:
            raise ValueError(f"{name}: projection differs")

    evidence_dim = int(arms[0][1]["local_evidence"].shape[1])
    template = LocalLogitResidual(
        evidence_dim, residual_cap=float(args.residual_cap)
    )
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in template.state_dict().items()
    }
    initial_sha = l89.state_dict_sha256(initial_state)
    results = {}
    for name, train, val in arms:
        results[name] = train_one_arm(
            name,
            train,
            val,
            args=args,
            initial_state=initial_state,
            output_dir=output_dir,
        )
    summary = {
        "format_version": HEAD_VERSION,
        "script_version": SCRIPT_VERSION,
        "protocol": "train_dev_only_frozen_final_patch_local_diagnostic",
        "selection_metric": "event_balanced_ap",
        "arms": results,
        "matching_audit": {
            "same_train_rows_labels_events": True,
            "same_val_rows_labels_events": True,
            "same_projection": True,
            "same_local_head_initial_state": True,
            "initial_state_sha256": initial_sha,
            "same_epoch_batches_and_optimizer": True,
            "local_head_parameters": int(
                sum(parameter.numel() for parameter in template.parameters())
            ),
            "local_head_bias": False,
            "static_cls_is_local_head_input": False,
            "base_logits_frozen": True,
            "epoch_zero_exact_base_logit": True,
            "backbone_trainable_parameters": 0,
            "token_layer": "final",
        },
        "configuration": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "residual_cap": float(args.residual_cap),
            "seed": int(args.seed),
            "train_loss_weighting": "event-balanced-times-class-balanced",
        },
        "interpretation_guardrail": (
            "Development-selected base checkpoint, local epoch, and threshold; "
            "diagnostic engineering evidence only, not a final test/SOTA result."
        ),
        "test_or_sealed_read": False,
    }
    atomic_json(output_dir / "comparison.json", summary)
    lines = [
        "# RCTP frozen final-patch local follow-up",
        "",
        "Train/development-only diagnostic; no test or sealed artifact was read.",
        "",
        "| Arm | Best epoch | Event AP base→best | Event macro-F1 base→best | Row AP base→best |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        zero = result["epoch_zero"]["validation"]
        best = result["best"]["validation"]
        lines.append(
            f"| {name} | {result['best']['epoch']} | "
            f"{zero['event_balanced_ap']:.4f}→{best['event_balanced_ap']:.4f} | "
            f"{zero['event_balanced_at_event_selected']['macro_f1']:.4f}→"
            f"{best['event_balanced_at_event_selected']['macro_f1']:.4f} | "
            f"{zero['row_ap']:.4f}→{best['row_ap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The local residual receives only same-position final-token "
            "`t0 - mean(valid unique history)` evidence. Static CLS is used "
            "only as a frozen base logit and cannot be relearned.",
            "",
        ]
    )
    (output_dir / "COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser(
        "extract", help="Extract pooled final-patch local evidence."
    )
    extract.add_argument("--arm", required=True)
    extract.add_argument("--split", choices=("train", "val"), required=True)
    extract.add_argument("--csv", required=True)
    extract.add_argument("--cls-cache", required=True)
    extract.add_argument("--weights", required=True)
    extract.add_argument("--base-head-checkpoint", required=True)
    extract.add_argument("--base-validation-predictions", default="")
    extract.add_argument("--output-cache", required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=12)
    extract.add_argument("--num-workers", type=int, default=4)
    extract.add_argument("--prefetch-factor", type=int, default=1)
    extract.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    extract.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    extract.add_argument("--projection-dim", type=int, default=64)
    extract.add_argument("--projection-seed", type=int, default=36064)
    extract.add_argument("--topk-fraction", type=float, default=0.10)
    extract.add_argument("--base-eval-batch-size", type=int, default=512)
    extract.add_argument("--min-cuda-free-gib", type=float, default=30.0)
    extract.add_argument("--max-cuda-allocated-gib", type=float, default=12.0)
    extract.add_argument(
        "--required-local-root",
        default=str(DEFAULT_CACHE_ROOT / "cache"),
    )
    extract.add_argument("--seed", type=int, default=20260728)
    extract.add_argument("--log-interval", type=int, default=25)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--debug", action="store_true")
    extract.set_defaults(handler=command_extract)

    train = commands.add_parser(
        "train-compare",
        help="Train matched zero-init local residual heads for all arms.",
    )
    train.add_argument(
        "--arm",
        nargs=3,
        action="append",
        metavar=("NAME", "TRAIN_CACHE", "VAL_CACHE"),
        required=True,
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--residual-cap", type=float, default=1.5)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=20260728)
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=command_train_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
