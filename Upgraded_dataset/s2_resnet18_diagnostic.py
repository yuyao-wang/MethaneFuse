#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


TIMEPOINT_COLUMNS = {
    "t0": "path_t0",
    "prev1": "path_prev1",
    "prev2": "path_prev2",
    "prev3": "path_prev3",
    "seasonal": "path_seasonal",
    "year": "path_year",
}

PATH_COLUMN_ALIASES = {
    "path_t0": ("image_path", "s2_path"),
    "path_prev1": ("s2_-7_path",),
    "path_seasonal": ("s2_pre_path",),
    "path_year": ("s2_pre_pre_path",),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    per_class = maximum // 2
    pieces = []
    for label in (0, 1):
        subset = frame[frame["label"].eq(label)]
        pieces.append(subset.sample(n=min(per_class, len(subset)), random_state=seed + label))
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_frame(
    csv_path: str,
    maximum: int,
    seed: int,
    center_box: int,
    path_columns: tuple[str, ...],
    mode: str,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    for target, aliases in PATH_COLUMN_ALIASES.items():
        if target in frame.columns:
            continue
        for alias in aliases:
            if alias in frame.columns:
                frame[target] = frame[alias]
                break
    required = {"label", *path_columns}
    if mode == "legacy_negative_prev1":
        required.add("path_prev1")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    if center_box > 0:
        if not {"crop_x", "crop_y"}.issubset(frame.columns):
            raise ValueError("--center-box requires crop_x and crop_y")
        lower = 256 - center_box // 2 - 16
        upper = 256 + center_box // 2 - 16
        keep_positive = frame["label"].eq(1)
        keep_negative = (
            frame["label"].eq(0)
            & frame["crop_x"].between(lower, upper)
            & frame["crop_y"].between(lower, upper)
        )
        frame = frame[keep_positive | keep_negative].copy()
    return balanced_sample(frame, maximum, seed)


def stage_frame_paths(
    frame: pd.DataFrame,
    path_columns: tuple[str, ...],
    cache_dir: Path,
    workers: int,
    split: str,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_to_destination: dict[str, Path] = {}
    for path_column in path_columns:
        for source_value in frame[path_column].astype(str):
            if source_value in source_to_destination:
                continue
            digest = hashlib.sha1(source_value.encode("utf-8")).hexdigest()
            suffix = Path(source_value).suffix or ".tif"
            shared_destination = cache_dir.parent / digest[:2] / f"{digest}{suffix}"
            destination = (
                shared_destination
                if shared_destination.is_file()
                else cache_dir / digest[:2] / f"{digest}{suffix}"
            )
            source_to_destination[source_value] = destination

    copied = 0
    reused = 0
    lock = threading.Lock()

    def stage_one(item: tuple[str, Path]) -> tuple[str, Path, bool]:
        source_value, destination = item
        source = Path(source_value)
        if destination.is_file() and cache_dir not in destination.parents:
            return source_value, destination, False
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return source_value, destination, False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}.part"
        )
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return source_value, destination, True

    total = len(source_to_destination)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(stage_one, item)
            for item in source_to_destination.items()
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            _, _, was_copied = future.result()
            with lock:
                if was_copied:
                    copied += 1
                else:
                    reused += 1
            if completed % 5000 == 0 or completed == total:
                print(
                    f"[Cache:{split}] {completed}/{total} copied={copied} reused={reused}",
                    flush=True,
                )

    staged = frame.copy()
    for path_column in path_columns:
        staged[path_column] = staged[path_column].astype(str).map(
            lambda value: str(source_to_destination[value])
        )
    return staged


def read_chw(path: str, band_indices: tuple[int, ...]) -> torch.Tensor:
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {array.shape}: {path}")
    if array.shape[0] not in {10, 12, 13} and array.shape[-1] in {10, 12, 13}:
        array = np.transpose(array, (2, 0, 1))
    if max(band_indices) >= array.shape[0]:
        raise ValueError(f"band selection {band_indices} exceeds {array.shape}: {path}")
    array = np.nan_to_num(array[list(band_indices)].astype(np.float32, copy=False))
    array = np.clip(array / 10000.0, 0.0, 2.0)
    return torch.from_numpy(np.ascontiguousarray(array))


class S2DiagnosticDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        mode: str,
        band_indices: tuple[int, ...],
        path_columns: tuple[str, ...],
    ) -> None:
        self.frame = frame
        self.mode = mode
        self.band_indices = band_indices
        self.path_columns = path_columns

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        label = int(row["label"])
        paths = []
        for path_column in self.path_columns:
            if self.mode == "legacy_negative_prev1" and label == 0 and path_column == "path_t0":
                path_column = "path_prev1"
            paths.append(row[path_column])
        image = torch.cat([read_chw(str(path), self.band_indices) for path in paths], dim=0)
        return image, label


def build_model(channels: int) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    true_positive = int(((labels_array == 1) & (predictions_array == 1)).sum())
    true_negative = int(((labels_array == 0) & (predictions_array == 0)).sum())
    false_positive = int(((labels_array == 0) & (predictions_array == 1)).sum())
    false_negative = int(((labels_array == 1) & (predictions_array == 0)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    accuracy = (true_positive + true_negative) / max(1, len(labels_array))
    auc = roc_auc_score(labels_array, np.asarray(scores)) if len(np.unique(labels_array)) == 2 else float("nan")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "auroc": float(auc),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    scores_all: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            probabilities = logits.softmax(dim=1)[:, 1]
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.detach()) * labels.numel()
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
            scores_all.extend(probabilities.detach().float().cpu().tolist())
    output = metrics(labels_all, predictions_all, scores_all)
    output["loss"] = total_loss / max(1, len(labels_all))
    return output


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    timepoints = tuple(value.strip() for value in args.timepoints.split(",") if value.strip())
    unknown_timepoints = sorted(set(timepoints) - set(TIMEPOINT_COLUMNS))
    if not timepoints or unknown_timepoints:
        raise ValueError(f"invalid --timepoints; unknown values: {unknown_timepoints}")
    path_columns = tuple(TIMEPOINT_COLUMNS[timepoint] for timepoint in timepoints)
    train_frame = prepare_frame(
        args.train_csv,
        args.max_train_samples,
        args.seed,
        args.center_box,
        path_columns,
        args.mode,
    )
    test_frame = prepare_frame(
        args.test_csv,
        args.max_test_samples,
        args.seed + 1000,
        args.center_box,
        path_columns,
        args.mode,
    )
    if args.local_cache_dir:
        cache_root = Path(args.local_cache_dir)
        train_frame = stage_frame_paths(
            train_frame,
            path_columns,
            cache_root / "train",
            args.cache_workers,
            "train",
        )
        test_frame = stage_frame_paths(
            test_frame,
            path_columns,
            cache_root / "test",
            args.cache_workers,
            "test",
        )
    band_indices = tuple(int(value) for value in args.band_indices.split(","))
    train_dataset = S2DiagnosticDataset(train_frame, args.mode, band_indices, path_columns)
    test_dataset = S2DiagnosticDataset(test_frame, args.mode, band_indices, path_columns)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    device = torch.device(args.device)
    model = build_model(len(path_columns) * len(band_indices)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "train_rows": len(train_frame),
                "test_rows": len(test_frame),
                "train_labels": train_frame["label"].value_counts().sort_index().to_dict(),
                "test_labels": test_frame["label"].value_counts().sort_index().to_dict(),
                "bands": band_indices,
                "timepoints": timepoints,
                "center_box": args.center_box,
                "device": str(device),
            }
        ),
        flush=True,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        test_metrics = run_epoch(model, test_loader, criterion, device, None)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 2),
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    output = {
        "args": vars(args),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "history": history,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument(
        "--mode",
        choices=("current", "legacy_negative_prev1"),
        default="current",
    )
    parser.add_argument("--timepoints", default="t0,seasonal,year")
    parser.add_argument("--band-indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--center-box", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-test-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--local-cache-dir", default="")
    parser.add_argument("--cache-workers", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="")
    main(parser.parse_args())
