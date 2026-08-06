#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.dino_clssifier_head_s2_temportal_one_block import (  # noqa: E402
    CLSHead,
    PRECOMPUTED_STATS,
)
from src.backbones import build_panopticon_vitb14  # noqa: E402
from thirdparty.dinov2.data.datasets.s2_csv import S2TemporalCsvDataset  # noqa: E402


def balanced_frame(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    per_class = maximum // 2
    pieces = []
    for label in (0, 1):
        subset = frame[pd.to_numeric(frame["label"], errors="raise").eq(label)]
        pieces.append(subset.sample(n=min(per_class, len(subset)), random_state=seed + label))
    return pd.concat(pieces, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)


class CachedTemporalDataset(S2TemporalCsvDataset):
    def __init__(
        self,
        *args,
        read_cache_dir: str = "",
        raw_dn_add: int = 0,
        raw_dn_add_columns: tuple[str, ...] = (),
        raw_dn_add_bands: tuple[int, ...] = (),
        **kwargs,
    ):
        self.read_cache_dir = Path(read_cache_dir) if read_cache_dir else None
        self.raw_dn_add = int(raw_dn_add)
        self.raw_dn_add_columns = frozenset(raw_dn_add_columns)
        self.raw_dn_add_bands = tuple(raw_dn_add_bands)
        super().__init__(*args, **kwargs)

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self.read_cache_dir is not None:
            digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
            candidate = self.read_cache_dir / digest[:2] / f"{digest}{Path(path).suffix}"
            if candidate.is_file():
                path = str(candidate)
        if (
            self.raw_dn_add
            and column_name in self.raw_dn_add_columns
            and self.raw_dn_add_bands
        ):
            image = self._read_image_raw(path)
            image = image.clone()
            band_indices = torch.tensor(self.raw_dn_add_bands, dtype=torch.long)
            selected = image.index_select(0, band_indices)
            selected = torch.where(
                selected > 0,
                torch.clamp(selected + self.raw_dn_add, min=0, max=65535),
                selected,
            )
            image.index_copy_(0, band_indices, selected)
            if self._mean is not None and self._std is not None:
                image = (image - self._mean) / self._std
            if self.pad_to_multiple is not None:
                image = self._pad_to_multiple(image, self.pad_to_multiple)
            return image
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)


class SelectedConcatTemporalDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        band_indices: tuple[int, ...],
        mask_band_indices: tuple[int, ...],
        legacy_t0_cdse_contract: bool = False,
    ):
        self.base_dataset = base_dataset
        self.band_indices = torch.tensor(band_indices, dtype=torch.long)
        self.legacy_t0_cdse_contract = legacy_t0_cdse_contract
        self.mean = torch.tensor(PRECOMPUTED_STATS[0], dtype=torch.float32).view(
            -1, 1, 1
        )
        self.std = torch.tensor(PRECOMPUTED_STATS[1], dtype=torch.float32).view(
            -1, 1, 1
        )
        self.mask_positions = tuple(
            position
            for position, band_index in enumerate(band_indices)
            if band_index in mask_band_indices
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        timepoints, label = self.base_dataset[index]
        if self.legacy_t0_cdse_contract:
            t0 = timepoints[0]["imgs"]
            source_t0 = t0 * self.std + self.mean
            raw_t0 = torch.zeros_like(source_t0)
            retained_channels = [0, 1, 2, 3, 4, 5, 6, 10, 11]
            retained = source_t0[retained_channels]
            raw_t0[retained_channels] = torch.where(
                retained.abs() < 0.5,
                torch.zeros_like(retained),
                retained + 1000,
            )
            legacy_b8a = source_t0[8]
            raw_t0[7] = torch.where(
                legacy_b8a.abs() < 0.5,
                torch.zeros_like(legacy_b8a),
                legacy_b8a + 1000,
            )
            timepoints[0]["imgs"] = (raw_t0 - self.mean) / self.std
        selected_images = []
        for timepoint in timepoints:
            image = timepoint["imgs"].index_select(0, self.band_indices)
            if self.mask_positions:
                image = image.clone()
                image[list(self.mask_positions)] = 0.0
            selected_images.append(image)
        images = torch.cat(selected_images, dim=0)
        channel_ids = torch.cat(
            [timepoint["chn_ids"].index_select(0, self.band_indices) for timepoint in timepoints],
            dim=0,
        )
        return {"imgs": images, "chn_ids": channel_ids}, label


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    f1_denominator = 2 * true_positive + false_positive + false_negative
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    f1_curve = np.divide(
        2 * precision_curve * recall_curve,
        precision_curve + recall_curve,
        out=np.zeros_like(precision_curve),
        where=(precision_curve + recall_curve) > 0,
    )
    best_index = int(np.argmax(f1_curve))
    best_threshold = (
        float(thresholds[best_index])
        if best_index < len(thresholds)
        else float(np.nextafter(scores.max(), np.inf))
    )
    return {
        "accuracy": float((predictions == labels).mean()),
        "f1": float(2 * true_positive / f1_denominator) if f1_denominator else 0.0,
        "best_test_f1": float(f1_curve[best_index]),
        "best_test_f1_threshold": best_threshold,
        "auroc": float(roc_auc_score(labels, scores)),
        "negative_score_mean": float(scores[labels == 0].mean()),
        "positive_score_mean": float(scores[labels == 1].mean()),
        "recall": float(true_positive / (true_positive + false_negative)),
        "fpr": float(false_positive / (false_positive + true_negative)),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path_columns = tuple(value.strip() for value in args.path_columns.split(","))
    band_indices = tuple(int(value) for value in args.band_indices.split(","))
    mask_band_indices = tuple(
        int(value)
        for value in args.mask_band_indices.split(",")
        if value.strip()
    )
    raw_dn_add_columns = tuple(
        value.strip()
        for value in args.raw_dn_add_columns.split(",")
        if value.strip()
    )
    raw_dn_add_bands = tuple(
        int(value)
        for value in args.raw_dn_add_bands.split(",")
        if value.strip()
    )
    dataset = CachedTemporalDataset(
        csv_path=args.csv,
        ds_cfg_name="s2_12band",
        normalize_stats=PRECOMPUTED_STATS,
        scale_to_unit=False,
        compute_stats=False,
        pad_to_multiple=None,
        path_columns=path_columns,
        read_cache_dir=args.read_cache_dir,
        raw_dn_add=args.raw_dn_add,
        raw_dn_add_columns=raw_dn_add_columns,
        raw_dn_add_bands=raw_dn_add_bands,
    )
    dataset.df = balanced_frame(dataset.df, args.max_samples, args.seed)
    selected_dataset = SelectedConcatTemporalDataset(
        dataset,
        band_indices,
        mask_band_indices,
        legacy_t0_cdse_contract=args.legacy_t0_cdse_contract,
    )
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone = build_panopticon_vitb14()
    backbone.load_state_dict(checkpoint["backbone"], strict=True)
    backbone = backbone.to(device).eval()
    head = CLSHead(embed_dim=768, num_classes=2)
    head.load_state_dict(checkpoint["head"], strict=True)
    head = head.to(device).eval()

    labels_all = []
    predictions_all = []
    scores_all = []
    with torch.inference_mode():
        for step, (inputs, labels) in enumerate(loader, start=1):
            images = inputs["imgs"].to(device, non_blocking=True)
            channel_ids = inputs["chn_ids"].to(device, non_blocking=True)
            images = F.interpolate(
                images,
                size=(args.input_size, args.input_size),
                mode="bilinear",
                align_corners=False,
            )
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda" and not args.disable_amp,
            ):
                features = backbone(
                    {"imgs": images, "chn_ids": channel_ids},
                    is_training=True,
                )
                logits = head(features["x_norm_clstoken"])
            probabilities = logits.softmax(dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            scores_all.extend(probabilities.float().cpu().numpy().tolist())
            if step % 100 == 0 or step == len(loader):
                print(f"[Eval] {step}/{len(loader)}", flush=True)

    labels_array = np.asarray(labels_all, dtype=np.int64)
    predictions_array = np.asarray(predictions_all, dtype=np.int64)
    scores_array = np.asarray(scores_all, dtype=np.float64)

    if args.save_scores:
        # Aggregate metrics hide how few independent samples some controlled
        # release sets contain, so keep the per-row scores for clustered
        # bootstrap and per-source breakdowns.
        import csv as _csv

        # Must follow dataset.df, not the raw CSV: balanced_frame() reshuffles
        # the frame, so the loader order is not the file order and re-reading
        # the CSV would pair each score with the wrong row.
        source_rows = dataset.df.to_dict("records")
        carry = [
            column
            for column in ("id", "plume_id", "label", "datetime", "latitude", "longitude")
            if source_rows and column in source_rows[0]
        ]
        out_path = Path(args.save_scores)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow([*carry, "eval_label", "score", "prediction"])
            for index in range(len(labels_array)):
                base = source_rows[index] if index < len(source_rows) else {}
                writer.writerow(
                    [
                        *[base.get(column, "") for column in carry],
                        int(labels_array[index]),
                        float(scores_array[index]),
                        int(predictions_array[index]),
                    ]
                )
        print(f"[Eval] wrote per-row scores to {out_path}", flush=True)
    report = {
        "csv": args.csv,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "rows": len(labels_array),
        "labels": {
            str(key): int(value)
            for key, value in zip(*np.unique(labels_array, return_counts=True))
        },
        "path_columns": path_columns,
        "band_indices": band_indices,
        "mask_band_indices": mask_band_indices,
        "raw_dn_add": args.raw_dn_add,
        "raw_dn_add_columns": raw_dn_add_columns,
        "raw_dn_add_bands": raw_dn_add_bands,
        "legacy_t0_cdse_contract": args.legacy_t0_cdse_contract,
        "metrics": binary_metrics(labels_array, predictions_array, scores_array),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-columns", required=True)
    parser.add_argument("--band-indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--mask-band-indices", default="")
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20261723)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--save-scores",
        default="",
        help="Optional CSV path for per-row scores (needed for clustered bootstrap).",
    )
    parser.add_argument("--read-cache-dir", default="")
    parser.add_argument("--raw-dn-add", type=int, default=0)
    parser.add_argument("--raw-dn-add-columns", default="")
    parser.add_argument("--raw-dn-add-bands", default="")
    parser.add_argument("--legacy-t0-cdse-contract", action="store_true")
    parser.add_argument("--output", default="")
    main(parser.parse_args())
