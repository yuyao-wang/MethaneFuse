#!/usr/bin/env python3
"""Frozen Copernicus-FM probe for six-date S5P methane NPZ crops.

This is a deliberately cheap foundation-model screen.  The methane variable
proxy is fixed before evaluation as the arithmetic mean of Copernicus-FM's
released S5P CO, NO2, SO2, and O3 language embeddings.  It is never selected on
validation performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset


SCRIPT_VERSION = 1
RESEARCH_ROOT = Path("/diniuvol/yuyao/methanefuse_research_20260727")
DEFAULT_REPO = RESEARCH_ROOT / "external/Copernicus-FM/Copernicus-FM"
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO / "weights/CopernicusFM_ViT_base_varlang_e100.pth"
)
DEFAULT_VARIABLE_EMBEDDINGS = (
    DEFAULT_REPO / "weights/varname_embed_llama3.2_1B.pt"
)
DEFAULT_TRAIN_CSV = RESEARCH_ROOT / "manifests_staged/s5p/train.csv"
DEFAULT_EVAL_CSV = RESEARCH_ROOT / "manifests_staged/s5p/val.csv"
DEFAULT_CACHE_ROOT = RESEARCH_ROOT / "cache/copernicusfm_s5p_probe"

PROXY_VARIABLE_KEYS = (
    "Sentinel 5P Carbon Monoxide",
    "Sentinel 5P Nitrogen Dioxide",
    "Sentinel 5P Sulfur Dioxide",
    "Sentinel 5P Ozone",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def selected_frame_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, ["_source_index", "image_path", "label", "plume_id"]]
    row_hashes = pd.util.hash_pandas_object(
        values.fillna("").astype(str), index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def balanced_selection(
    frame: pd.DataFrame,
    limit: int,
    *,
    seed: int,
) -> pd.DataFrame:
    if limit <= 0 or limit % 2:
        raise ValueError("Balanced selection limit must be a positive even integer")
    per_class = limit // 2
    selected = []
    for label in (0, 1):
        group = frame[frame["label"] == label]
        if len(group) < per_class:
            raise ValueError(
                f"Need {per_class} rows for label={label}, found {len(group)}"
            )
        selected.append(
            group.sample(n=per_class, random_state=seed + label, replace=False)
        )
    return (
        pd.concat(selected)
        .sort_values("_source_index", kind="stable")
        .reset_index(drop=True)
    )


def read_manifests(
    train_csv: Path,
    eval_csv: Path,
    *,
    max_train: int,
    max_eval: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = ("image_path", "label", "plume_id")
    frames = []
    for split, path in (("train", train_csv), ("eval", eval_csv)):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=list(required), low_memory=False)
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{split} manifest contains missing required values")
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(
            np.int64
        )
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{split} labels are not binary")
        frame["_source_index"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)

    train_full, eval_full = frames
    overlap = set(train_full["plume_id"].astype(str)) & set(
        eval_full["plume_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"Train/eval plume_id overlap ({len(overlap)}), examples={sorted(overlap)[:10]}"
        )

    train = balanced_selection(train_full, max_train, seed=seed)
    evaluation = balanced_selection(eval_full, max_eval, seed=seed + 10_000)
    audit = {
        "train_csv": str(train_csv.expanduser().resolve()),
        "eval_csv": str(eval_csv.expanduser().resolve()),
        "full_rows": {"train": len(train_full), "eval": len(eval_full)},
        "selected_rows": {"train": len(train), "eval": len(evaluation)},
        "selected_labels": {
            "train": {
                str(key): int(value)
                for key, value in train["label"].value_counts().sort_index().items()
            },
            "eval": {
                str(key): int(value)
                for key, value in evaluation["label"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "selected_unique_plumes": {
            "train": int(train["plume_id"].nunique()),
            "eval": int(evaluation["plume_id"].nunique()),
        },
        "plume_id_overlap": 0,
        "selection_seed": seed,
        "selection": "exactly half of each label, deterministic pandas sample",
    }
    return train, evaluation, audit


def load_ch4(path: str) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        array = np.asarray(archive["ch4"], dtype=np.float32)
    if array.shape != (6, 224, 224):
        raise ValueError(f"{source}: expected (6,224,224), got {array.shape}")
    return np.ascontiguousarray(array)


def one_file_stats(path: str) -> tuple[int, float, float]:
    array = load_ch4(path)
    finite = np.isfinite(array)
    values = array[finite].astype(np.float64, copy=False)
    return int(values.size), float(values.sum()), float(np.square(values).sum())


def compute_stats(
    paths: list[str],
    *,
    workers: int,
    progress_every: int,
) -> dict[str, Any]:
    count = 0
    total = 0.0
    total_sq = 0.0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, (item_count, item_sum, item_sq) in enumerate(
            executor.map(one_file_stats, paths), 1
        ):
            count += item_count
            total += item_sum
            total_sq += item_sq
            if progress_every and (
                index % progress_every == 0 or index == len(paths)
            ):
                print(
                    f"[stats] {index}/{len(paths)} files "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    if count == 0:
        raise ValueError("No finite S5P training values")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not math.isfinite(std) or std <= 1e-8:
        raise ValueError(f"Invalid S5P training std={std}")
    return {
        "mean": mean,
        "std": std,
        "finite_count": count,
        "files": len(paths),
    }


def load_or_compute_stats(
    train: pd.DataFrame,
    run_root: Path,
    *,
    workers: int,
    progress_every: int,
    recompute: bool,
) -> tuple[dict[str, Any], Path]:
    path = run_root / "train_stats.json"
    if path.is_file() and not recompute:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"[stats] reusing {path}", flush=True)
        return payload["stats"], path
    stats = compute_stats(
        train["image_path"].astype(str).tolist(),
        workers=workers,
        progress_every=progress_every,
    )
    atomic_json(
        path,
        {
            "source": "selected training rows only",
            "selection_hash": selected_frame_hash(train),
            "stats": stats,
            "created_unix": time.time(),
        },
    )
    print(
        f"[stats] mean={stats['mean']:.6f} std={stats['std']:.6f} wrote={path}",
        flush=True,
    )
    return stats, path


class S5PNpzDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.paths = frame["image_path"].astype(str).tolist()
        self.labels = frame["label"].to_numpy(dtype=np.int64)
        self.source_indices = frame["_source_index"].to_numpy(dtype=np.int64)
        self.plume_ids = frame["plume_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, int, int, str]:
        try:
            array = load_ch4(self.paths[index])
        except Exception as exc:
            raise RuntimeError(
                f"Failed NPZ sample index={index}, plume_id={self.plume_ids[index]}"
            ) from exc
        return (
            torch.from_numpy(array[:, None, :, :]),
            int(self.labels[index]),
            int(self.source_indices[index]),
            self.plume_ids[index],
        )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if workers > 0 else None,
    )


def load_proxy_embedding(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=True
    )
    missing = [key for key in PROXY_VARIABLE_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Missing fixed proxy embeddings: {missing}")
    values = [payload[key].to(torch.float32) for key in PROXY_VARIABLE_KEYS]
    if any(tuple(value.shape) != (2048,) for value in values):
        raise ValueError("Expected every variable language embedding to be 2048-D")
    proxy = torch.stack(values, dim=0).mean(dim=0)
    identity = {
        "policy": "predeclared arithmetic mean; never validation-selected",
        "keys": list(PROXY_VARIABLE_KEYS),
        "source": file_identity(path),
        "dimension": int(proxy.numel()),
        "component_norms": [float(value.norm().item()) for value in values],
        "proxy_norm": float(proxy.norm().item()),
    }
    return proxy.to(device), identity


def load_backbone(
    repo: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    repo = repo.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    if not (repo / "src/model_vit.py").is_file():
        raise FileNotFoundError(repo / "src/model_vit.py")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # The official source mixes ``src.*`` relative imports with top-level
    # ``util.*`` imports, so both roots are required for its unmodified code.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    from src.model_vit import vit_base_patch16

    model = vit_base_patch16(num_classes=0, global_pool=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload["model"] if "model" in payload else payload
    message = model.load_state_dict(state_dict, strict=False)
    model.requires_grad_(False)
    model.eval().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    identity = {
        "repo": str(repo),
        "checkpoint": file_identity(checkpoint),
        "parameters": int(parameters),
        "architecture": "official vit_base_patch16, global_pool=False",
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
    }
    print(
        f"[model] loaded {parameters:,} frozen parameters; "
        f"missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}",
        flush=True,
    )
    return model, identity


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.no_grad()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


def extract_features(
    split: str,
    frame: pd.DataFrame,
    *,
    stats: dict[str, Any],
    model: nn.Module,
    language_proxy: torch.Tensor,
    model_identity: dict[str, Any],
    proxy_identity: dict[str, Any],
    run_root: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp_dtype: str,
    progress_every: int,
) -> Path:
    cache_path = run_root / f"{split}_features.pt"
    config = {
        "script_version": SCRIPT_VERSION,
        "split": split,
        "selection_hash": selected_frame_hash(frame),
        "rows": len(frame),
        "stats": stats,
        "model": model_identity,
        "proxy": proxy_identity,
        "resize": [56, 56],
        "input_mode": "variable",
        "kernel_size": 4,
        "metadata": "all NaN / official unknown-token path",
    }
    key = canonical_hash(config)
    if cache_path.is_file():
        cached = torch.load(
            cache_path, map_location="cpu", weights_only=True
        )
        if cached.get("feature_key") == key:
            print(
                f"[features:{split}] reusing {len(cached['labels'])} rows from {cache_path}",
                flush=True,
            )
            return cache_path

    dataset = S5PNpzDataset(frame)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        pin_memory=device.type == "cuda",
    )
    output_features = []
    output_labels = []
    output_indices = []
    mean = float(stats["mean"])
    std = float(stats["std"])
    started = time.time()
    for step, (images, labels, indices, _plume_ids) in enumerate(loader, 1):
        batch, times = images.shape[:2]
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        images = torch.nan_to_num(images, nan=mean, posinf=mean, neginf=mean)
        images = (images - mean) / std
        images = images.flatten(0, 1)
        images = F.interpolate(
            images, size=(56, 56), mode="bilinear", align_corners=False
        )
        metadata = torch.full(
            (batch * times, 4),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode(), autocast_context(device, amp_dtype):
            embedding = model.forward_features(
                images,
                metadata,
                None,
                None,
                language_proxy,
                "variable",
                4,
            )
        if embedding.ndim != 2 or embedding.shape[0] != batch * times:
            raise ValueError(
                f"Unexpected backbone output shape {tuple(embedding.shape)}"
            )
        output_features.append(
            embedding.reshape(batch, times, -1).to(torch.float16).cpu()
        )
        output_labels.append(labels.to(torch.int64))
        output_indices.append(indices.to(torch.int64))
        if progress_every and (
            step % progress_every == 0 or step == len(loader)
        ):
            print(
                f"[features:{split}] batch={step}/{len(loader)} "
                f"rows={min(step * batch_size, len(dataset))}/{len(dataset)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    payload = {
        "feature_key": key,
        "config": config,
        "features": torch.cat(output_features, dim=0),
        "labels": torch.cat(output_labels, dim=0),
        "source_indices": torch.cat(output_indices, dim=0),
    }
    if len(payload["labels"]) != len(frame):
        raise RuntimeError("Extracted feature row count mismatch")
    atomic_torch_save(cache_path, payload)
    print(f"[features:{split}] wrote {cache_path}", flush=True)
    return cache_path


class MeanMaxTemporalHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim * 2)
        self.classifier = nn.Linear(feature_dim * 2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [features.mean(dim=1), features.amax(dim=1)], dim=-1
        )
        return self.classifier(self.norm(pooled))


def best_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        predictions = probabilities >= 0.5
        return 0.5, float(f1_score(labels, predictions, zero_division=0))
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    best = float(scores.max())
    candidates = np.flatnonzero(np.isclose(scores, best))
    index = min(
        candidates.tolist(),
        key=lambda item: (abs(float(thresholds[item]) - 0.5), item),
    )
    return float(thresholds[index]), best


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    threshold, best_f1 = best_f1_threshold(labels, probabilities)
    predictions = (probabilities >= threshold).astype(np.int64)
    predictions_05 = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "rows": int(len(labels)),
        "ap": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "best_threshold": threshold,
        "f1": best_f1,
        "f1_at_0_5": float(
            f1_score(labels, predictions_05, zero_division=0)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probabilities = []
    targets = []
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            logits = model(
                batch_features.to(device, dtype=torch.float32, non_blocking=True)
            )
            probabilities.append(
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
            targets.append(batch_labels.numpy())
    return compute_metrics(
        np.concatenate(targets).astype(np.int64, copy=False),
        np.concatenate(probabilities).astype(np.float64, copy=False),
    )


def train_head(
    train_cache: Path,
    eval_cache: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    result_path: Path,
    provenance: dict[str, Any],
    promotion_ap: float,
) -> dict[str, Any]:
    train = torch.load(train_cache, map_location="cpu", weights_only=True)
    evaluation = torch.load(eval_cache, map_location="cpu", weights_only=True)
    train_features = train["features"]
    train_labels = train["labels"]
    eval_features = evaluation["features"]
    eval_labels = evaluation["labels"]
    if train_features.ndim != 3 or train_features.shape[1] != 6:
        raise ValueError(f"Unexpected train feature shape {train_features.shape}")
    if eval_features.shape[1:] != train_features.shape[1:]:
        raise ValueError("Train/eval feature shapes differ")

    head = MeanMaxTemporalHead(int(train_features.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    history = []
    best = None
    best_state = None
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_rows = 0
        for features, labels in train_loader:
            features = features.to(
                device, dtype=torch.float32, non_blocking=True
            )
            labels = labels.to(device, non_blocking=True)
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(labels.numel())
            total_loss += float(loss.item()) * rows
            total_rows += rows
        metrics = evaluate(
            head,
            eval_features,
            eval_labels,
            batch_size=batch_size,
            device=device,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "eval": metrics,
        }
        history.append(record)
        if best is None or (
            metrics["ap"], metrics["f1"]
        ) > (
            best["eval"]["ap"], best["eval"]["f1"]
        ):
            best = record
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
        print(
            f"[head] epoch={epoch}/{epochs} loss={record['train_loss']:.6f} "
            f"AP={metrics['ap']:.6f} AUROC={metrics['auroc']:.6f} "
            f"F1={metrics['f1']:.6f} F1@0.5={metrics['f1_at_0_5']:.6f}",
            flush=True,
        )
        atomic_json(
            result_path,
            {
                "status": "running",
                **provenance,
                "history": history,
                "best": best,
            },
        )

    if best is None or best_state is None:
        raise RuntimeError("No head epoch completed")
    decision = (
        "stop_below_panopticon_baseline"
        if best["eval"]["ap"] < promotion_ap
        else "eligible_for_parent_review"
    )
    checkpoint_path = result_path.with_name(
        result_path.stem + "_head_best.pt"
    )
    atomic_torch_save(
        checkpoint_path,
        {
            "head": best_state,
            "best_epoch": int(best["epoch"]),
            "feature_dim": int(train_features.shape[-1]),
        },
    )
    result = {
        "status": "complete",
        **provenance,
        "history": history,
        "best": best,
        "screen_decision": {
            "panopticon_validation_ap": promotion_ap,
            "copernicusfm_best_ap": best["eval"]["ap"],
            "difference": best["eval"]["ap"] - promotion_ap,
            "decision": decision,
            "full_run_launched": False,
        },
        "head_checkpoint": str(checkpoint_path),
    }
    atomic_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval_csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--variable_embeddings",
        type=Path,
        default=DEFAULT_VARIABLE_EMBEDDINGS,
    )
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_train", type=int, default=2048)
    parser.add_argument("--max_eval", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--stats_workers", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--head_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--amp_dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--promotion_ap", type=float, default=0.644)
    parser.add_argument("--recompute_stats", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_train",
        "max_eval",
        "stats_workers",
        "extract_batch_size",
        "head_batch_size",
        "epochs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.max_train % 2 or args.max_eval % 2:
        raise ValueError("--max_train and --max_eval must be even")
    if not 0 <= args.num_workers <= 3:
        raise ValueError("--num_workers must be between 0 and 3")
    if not 1 <= args.stats_workers <= 3:
        raise ValueError("--stats_workers must be between 1 and 3")
    if args.epochs > 3:
        raise ValueError("This screen is capped at three head epochs")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    train, evaluation, manifest_audit = read_manifests(
        args.train_csv,
        args.eval_csv,
        max_train=args.max_train,
        max_eval=args.max_eval,
        seed=args.seed,
    )
    run_config = {
        "script_version": SCRIPT_VERSION,
        "train_selection": selected_frame_hash(train),
        "eval_selection": selected_frame_hash(evaluation),
        "checkpoint": file_identity(args.checkpoint),
        "variable_embeddings": file_identity(args.variable_embeddings),
        "proxy_keys": list(PROXY_VARIABLE_KEYS),
        "seed": args.seed,
        "max_train": args.max_train,
        "max_eval": args.max_eval,
    }
    run_key = canonical_hash(run_config)
    run_root = args.cache_root.expanduser().resolve() / run_key[:16]
    output_json = args.output_json.expanduser().resolve()
    atomic_json(
        output_json,
        {
            "status": "starting",
            "run_key": run_key,
            "run_root": str(run_root),
            "command": [sys.executable, *sys.argv],
            "args": json_safe(vars(args)),
            "manifest_audit": manifest_audit,
        },
    )

    stats, stats_path = load_or_compute_stats(
        train,
        run_root,
        workers=args.stats_workers,
        progress_every=args.progress_every,
        recompute=args.recompute_stats,
    )
    proxy, proxy_identity = load_proxy_embedding(
        args.variable_embeddings, device
    )
    model, model_identity = load_backbone(
        args.repo, args.checkpoint, device
    )
    train_cache = extract_features(
        "train",
        train,
        stats=stats,
        model=model,
        language_proxy=proxy,
        model_identity=model_identity,
        proxy_identity=proxy_identity,
        run_root=run_root,
        device=device,
        batch_size=args.extract_batch_size,
        workers=args.num_workers,
        amp_dtype=args.amp_dtype,
        progress_every=args.progress_every,
    )
    eval_cache = extract_features(
        "eval",
        evaluation,
        stats=stats,
        model=model,
        language_proxy=proxy,
        model_identity=model_identity,
        proxy_identity=proxy_identity,
        run_root=run_root,
        device=device,
        batch_size=args.extract_batch_size,
        workers=args.num_workers,
        amp_dtype=args.amp_dtype,
        progress_every=args.progress_every,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    provenance = {
        "run_key": run_key,
        "run_root": str(run_root),
        "command": [sys.executable, *sys.argv],
        "args": json_safe(vars(args)),
        "manifest_audit": manifest_audit,
        "training_stats": {
            "source": "selected balanced training rows only",
            "path": str(stats_path),
            **stats,
        },
        "proxy_embedding": proxy_identity,
        "backbone": model_identity,
        "feature_cache": {
            "train": str(train_cache),
            "eval": str(eval_cache),
        },
        "temporal_head": "per-frame frozen embedding; concatenate temporal mean and max; LayerNorm + Linear(2)",
        "metadata_policy": "all NaN, invoking official unknown metadata tokens",
    }
    result = train_head(
        train_cache,
        eval_cache,
        epochs=args.epochs,
        batch_size=args.head_batch_size,
        learning_rate=args.head_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        result_path=output_json,
        provenance=provenance,
        promotion_ap=args.promotion_ap,
    )
    best = result["best"]["eval"]
    print(
        f"[done] AP={best['ap']:.6f} AUROC={best['auroc']:.6f} "
        f"F1={best['f1']:.6f} decision={result['screen_decision']['decision']} "
        f"output={output_json}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        if "--output_json" in sys.argv:
            index = sys.argv.index("--output_json")
            if index + 1 < len(sys.argv):
                failure_path = Path(sys.argv[index + 1]).expanduser().resolve()
                failure_payload: dict[str, Any] = {}
                if failure_path.is_file():
                    try:
                        failure_payload = json.loads(
                            failure_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        failure_payload = {}
                failure_payload.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_unix": time.time(),
                    }
                )
                atomic_json(failure_path, failure_payload)
        raise
