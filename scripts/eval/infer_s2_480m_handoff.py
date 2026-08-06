#!/usr/bin/env python
"""Preprocess the 75-row S2 handoff with legacy code and run 480 m inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.evaluate_classification import json_safe
from src.data.multisensor import TriSensorTemporalCsvDataset, custom_collate_fn
from src.data.sensor_transforms import DEFAULT_WV3_BANDS, load_wv3_channel_ids_from_srf
from src.evaluation.metrics import compute_split_metrics
from src.models.finetune_loramoe_adapter import MultiSensorPanopticonClassifier
from src.utils.training import _load_backbone, load_model_checkpoint_flexible, recursive_to_device

DEFAULT_HANDOFF_ROOT = Path("/diniuvol/yuyao/methanefuse_handoff_75rows")
DEFAULT_LEGACY_MODULE = Path(
    "/home/yuyao/methane_train/preprocess_dataset_query_multi/crop_legacy_param.py"
)
DEFAULT_CHECKPOINT = Path(
    "/transferdiniu2/yuyao/checkpoints/universal_480m/query dataset 480m/ckpt_best_test.pth"
)


def load_legacy_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Legacy preprocessing module not found: {path}")
    spec = importlib.util.spec_from_file_location("methane_train_crop_legacy_param", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import legacy preprocessing module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_source_images(input_dir: Path) -> dict[str, Path]:
    suffix = "_S2_12band_smoke.tif"
    resolved: dict[str, Path] = {}
    for path in sorted(input_dir.glob(f"*{suffix}")):
        name = path.name
        if len(name) < 6 or name[4] != "_":
            continue
        sample_id = name[5 : -len(suffix)]
        if sample_id in resolved:
            raise RuntimeError(f"Duplicate image mapping for id={sample_id}: {path}")
        resolved[sample_id] = path
    return resolved


def read_chw(legacy, path: Path) -> np.ndarray:
    arr = legacy.tifffile.imread(str(path))
    chw = legacy.to_chw(np.asarray(arr))
    if chw is None:
        raise ValueError(f"Unsupported TIFF shape for {path}: {arr.shape}")
    if chw.shape[0] != 12:
        raise ValueError(f"Expected 12 S2 bands for {path}, got shape={chw.shape}")
    return chw


def atomic_tiff_write(legacy, path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.tif")
    try:
        legacy.tifffile.imwrite(str(tmp), arr)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def preprocess(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    legacy = load_legacy_module(args.legacy_module)
    source_df = pd.read_csv(args.input_csv)
    if args.limit > 0:
        source_df = source_df.head(args.limit).copy()
    required = {"id", "label", "s2_0_path", "s2_90_path", "s2_360_path"}
    missing = sorted(required.difference(source_df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    source_images = resolve_source_images(args.input_dir)
    crop_px = int(round(args.query_size_m / args.s2_gsd_m))
    if crop_px <= 0:
        raise ValueError(f"Invalid crop size: {crop_px}")

    output_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row_idx, row in source_df.iterrows():
        sample_id = str(row["id"])
        try:
            source_path = source_images[sample_id]
        except KeyError as exc:
            raise FileNotFoundError(f"No local smoke TIFF found for id={sample_id}") from exc

        input_arr = read_chw(legacy, source_path)
        _, height, width = input_arr.shape
        if height < crop_px or width < crop_px:
            raise ValueError(
                f"Input is smaller than the requested {crop_px}px crop: "
                f"{source_path} has shape={input_arr.shape}"
            )
        x = (width - crop_px) // 2
        y = (height - crop_px) // 2
        cropped = legacy.crop_chw(input_arr, x, y, crop_px)
        if cropped is None or cropped.shape != (12, crop_px, crop_px):
            raise RuntimeError(
                f"Legacy center crop failed for {source_path}: "
                f"expected={(12, crop_px, crop_px)}, got={None if cropped is None else cropped.shape}"
            )
        resized = legacy.resize_chw(cropped, args.target_size)
        if resized.shape != (12, args.target_size, args.target_size):
            raise RuntimeError(f"Legacy resize returned unexpected shape for {source_path}: {resized.shape}")
        resized = resized.astype(np.float32, copy=False)

        output_path = args.output_dir / source_path.name.replace(
            "_S2_12band_smoke.tif", f"_S2_{int(args.query_size_m)}m_{args.target_size}.tif"
        )
        if args.overwrite or not output_path.exists():
            atomic_tiff_write(legacy, output_path, resized)
        check = read_chw(legacy, output_path)
        if check.shape != resized.shape or check.dtype != np.float32:
            raise RuntimeError(
                f"Output verification failed for {output_path}: "
                f"shape={check.shape}, dtype={check.dtype}"
            )

        out_row = row.to_dict()
        for column in ("s2_0_path", "s2_90_path", "s2_360_path"):
            out_row[column] = str(output_path)
        output_rows.append(out_row)
        records.append(
            {
                "id": sample_id,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "input_shape": list(input_arr.shape),
                "crop_xy": [int(x), int(y)],
                "crop_shape": list(cropped.shape),
                "output_shape": list(resized.shape),
                "output_dtype": str(resized.dtype),
            }
        )
        if len(output_rows) % 10 == 0 or len(output_rows) == len(source_df):
            print(f"[preprocess] {len(output_rows)}/{len(source_df)}", flush=True)

    output_df = pd.DataFrame(output_rows)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_manifest, index=False)
    report = {
        "legacy_module": str(args.legacy_module),
        "legacy_functions": ["crop_chw", "resize_chw"],
        "input_csv": str(args.input_csv),
        "input_dir": str(args.input_dir),
        "output_manifest": str(args.output_manifest),
        "count": len(output_df),
        "query_size_m": float(args.query_size_m),
        "s2_gsd_m": float(args.s2_gsd_m),
        "crop_size_px": crop_px,
        "target_size": int(args.target_size),
        "interpolation": "legacy torch bilinear, align_corners=False",
        "samples": records,
    }
    args.preprocess_report.parent.mkdir(parents=True, exist_ok=True)
    args.preprocess_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_df, report


def binary_details(labels: list[int], preds: list[int]) -> dict[str, Any]:
    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    tp = int(((labels_np == 1) & (preds_np == 1)).sum())
    fp = int(((labels_np == 0) & (preds_np == 1)).sum())
    tn = int(((labels_np == 0) & (preds_np == 0)).sum())
    fn = int(((labels_np == 1) & (preds_np == 0)).sum())

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * precision * recall, precision + recall)
        if math.isfinite(precision + recall) and (precision + recall) > 0
        else float("nan"),
    }


def infer(args: argparse.Namespace, manifest_df: pd.DataFrame) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    wv3_chn_ids = load_wv3_channel_ids_from_srf(
        str(args.wv3_srf_csv), DEFAULT_WV3_BANDS
    ).unsqueeze(-1)
    dataset = TriSensorTemporalCsvDataset(
        csv_path=str(args.output_manifest),
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=custom_collate_fn,
    )

    # Load the complete training checkpoint on CPU first. This avoids placing
    # optimizer/scheduler tensors on a memory-constrained shared GPU.
    model = MultiSensorPanopticonClassifier(
        backbone=_load_backbone(str(args.weights)),
        enable_summary_head=False,
        row_fusion_mode="max",
    )
    load_model_checkpoint_flexible(args.checkpoint, model, torch.device("cpu"))
    model = model.to(device)
    model.eval()

    labels_all: list[int] = []
    preds_all: list[int] = []
    scores_all: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    row_cursor = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for step, (x_dict, labels, sensors, sample_to_row) in enumerate(loader, 1):
            batch_size = int(labels.shape[0])
            expected = manifest_df.iloc[row_cursor : row_cursor + batch_size]
            expected_labels = expected["label"].astype(int).tolist()
            if expected_labels != labels.tolist():
                raise RuntimeError(
                    "Dataset row order changed during inference; refusing to attach predictions "
                    "to potentially incorrect ids."
                )

            x_dict = recursive_to_device(x_dict, device)
            sample_to_row = sample_to_row.to(device=device, dtype=torch.long)
            with autocast(enabled=use_amp):
                outputs = model(x_dict, sensors=sensors)
                fused_logits = model.compute_row_fused_logits(
                    outputs,
                    sample_to_row=sample_to_row,
                    num_rows=batch_size,
                    device=device,
                )
            fused_probs = torch.softmax(fused_logits.float(), dim=1)
            batch_preds = fused_probs.argmax(dim=1).cpu().tolist()
            batch_scores = fused_probs[:, 1].cpu().tolist()

            s2_scores = torch.full((batch_size,), float("nan"), device=device)
            s2_output = outputs.get("s2")
            if isinstance(s2_output, dict) and s2_output["indices"].numel() > 0:
                sensor_prob = torch.softmax(s2_output["logits"].float(), dim=1)[:, 1]
                sensor_rows = sample_to_row.index_select(0, s2_output["indices"])
                s2_scores.index_copy_(0, sensor_rows, sensor_prob)
            batch_s2_scores = s2_scores.cpu().tolist()

            for local_idx, (_, source_row) in enumerate(expected.iterrows()):
                label = int(source_row["label"])
                pred = int(batch_preds[local_idx])
                rec = source_row.to_dict()
                rec.update(
                    {
                        "probability_class_1": float(batch_scores[local_idx]),
                        "prediction": pred,
                        "correct": bool(pred == label),
                        "s2_head_probability_class_1": float(batch_s2_scores[local_idx]),
                    }
                )
                prediction_rows.append(rec)
            labels_all.extend(expected_labels)
            preds_all.extend(batch_preds)
            scores_all.extend(float(x) for x in batch_scores)
            row_cursor += batch_size
            print(
                f"[inference] batch={step}/{len(loader)} rows={row_cursor}/{len(dataset)}",
                flush=True,
            )

    if row_cursor != len(manifest_df):
        raise RuntimeError(f"Inference count mismatch: predicted={row_cursor}, expected={len(manifest_df)}")

    predictions_df = pd.DataFrame(prediction_rows)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.predictions_csv, index=False)

    overall = compute_split_metrics(labels_all, preds_all, scores_all)
    overall.update(binary_details(labels_all, preds_all))
    by_site: dict[str, Any] = {}
    if "site_id" in predictions_df.columns:
        for site_id, group in predictions_df.groupby("site_id", dropna=False):
            site_labels = group["label"].astype(int).tolist()
            site_preds = group["prediction"].astype(int).tolist()
            site_scores = group["probability_class_1"].astype(float).tolist()
            site_metrics = compute_split_metrics(site_labels, site_preds, site_scores)
            site_metrics.update(binary_details(site_labels, site_preds))
            by_site[str(site_id)] = site_metrics

    result = json_safe(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_contract": {
                "stage": "a",
                "row_fusion_mode": "max",
                "primary_output": "row_fusion_head",
            },
            "manifest": str(args.output_manifest),
            "predictions_csv": str(args.predictions_csv),
            "count": len(labels_all),
            "positive_labels": int(sum(labels_all)),
            "predicted_positive": int(sum(preds_all)),
            "overall": overall,
            "by_site": by_site,
        }
    )
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--preprocess-report", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--legacy-module", type=Path, default=DEFAULT_LEGACY_MODULE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "weights" / "panopticon_vitb14_teacher.pth"
    )
    parser.add_argument(
        "--wv3-srf-csv",
        type=Path,
        default=REPO_ROOT / "data" / "manifests" / "WV3_VNIR_SWIR_response.csv",
    )
    parser.add_argument("--query-size-m", type=float, default=480.0)
    parser.add_argument("--s2-gsd-m", type=float, default=10.0)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    args.input_csv = args.input_csv or args.handoff_root / "52_methanefuse_smoke_test.csv"
    args.input_dir = args.input_dir or args.handoff_root / "s2_12band_smoke"
    args.output_dir = args.output_dir or args.handoff_root / "s2_480m_224"
    args.output_manifest = (
        args.output_manifest or args.handoff_root / "52_methanefuse_smoke_test_480m_224.csv"
    )
    args.preprocess_report = (
        args.preprocess_report or args.handoff_root / "preprocess_480m_224_report.json"
    )
    args.predictions_csv = (
        args.predictions_csv or args.handoff_root / "classification_480m_predictions.csv"
    )
    args.metrics_json = (
        args.metrics_json or args.handoff_root / "classification_480m_metrics.json"
    )
    return args


def main() -> None:
    args = parse_args()
    if args.skip_preprocess:
        manifest_df = pd.read_csv(args.output_manifest)
    else:
        manifest_df, _ = preprocess(args)
    if not args.preprocess_only:
        infer(args, manifest_df)


if __name__ == "__main__":
    main()
