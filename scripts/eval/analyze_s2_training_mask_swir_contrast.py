#!/usr/bin/env python3
"""Measure whether training-positive masks have observable S2 SWIR contrast."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-reference-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def standardized_contrast(values: np.ndarray, mask: np.ndarray) -> float | None:
    ring = binary_dilation(mask, iterations=28) & ~binary_dilation(mask, iterations=3)
    if int(mask.sum()) < 16 or int(ring.sum()) < 64:
        return None
    inside = values[mask]
    background = values[ring]
    std = float(background.std())
    if std <= 1e-8:
        return None
    return float((inside.mean() - background.mean()) / std)


def one_row(record: dict[str, object]) -> dict[str, object]:
    image = np.asarray(tifffile.imread(str(record["s2_0_path"])), dtype=np.float32)
    mask = np.asarray(tifffile.imread(str(record["s2_plume_path"])))
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    plume = mask > 0.5
    # A deterministic large wraparound shift supplies a same-image null location.
    shifted = np.roll(np.roll(plume, 73, axis=0), 91, axis=1)
    b11 = image[10].astype(np.float64)
    b12 = image[11].astype(np.float64)
    swir_nd = (b11 - b12) / np.maximum(b11 + b12, 1.0)
    visible = image[:4].mean(axis=0, dtype=np.float64)
    output: dict[str, object] = {
        "id": str(record["id"]),
        "plume_id": str(record["plume_id"]),
        "mask_pixels": int(plume.sum()),
    }
    for name, values in (("swir_nd", swir_nd), ("b11", b11), ("b12", b12), ("visible", visible)):
        output[f"{name}_true_contrast"] = standardized_contrast(values, plume)
        output[f"{name}_shifted_contrast"] = standardized_contrast(values, shifted)
    return output


def signed_summary(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    true = frame[f"{prefix}_true_contrast"].to_numpy(dtype=float)
    shifted = frame[f"{prefix}_shifted_contrast"].to_numpy(dtype=float)
    valid = np.isfinite(true) & np.isfinite(shifted)
    true = true[valid]
    shifted = shifted[valid]
    return {
        "count": len(true),
        "true_mean": float(true.mean()),
        "true_median": float(np.median(true)),
        "true_fraction_positive": float(np.mean(true > 0)),
        "true_fraction_abs_gt_0p5": float(np.mean(np.abs(true) > 0.5)),
        "shifted_mean": float(shifted.mean()),
        "shifted_median": float(np.median(shifted)),
        "paired_true_minus_shifted_mean": float((true - shifted).mean()),
        "paired_true_vs_shifted_p": float(wilcoxon(true, shifted).pvalue),
        "true_vs_zero_p": float(wilcoxon(true).pvalue),
    }


def main() -> None:
    args = parse_args()
    reference = json.loads(args.training_reference_json.read_text(encoding="utf-8"))
    train = pd.read_csv(reference["training_manifest"], low_memory=False)
    train["id"] = train["id"].astype(str)
    ids = [str(value) for value in reference["sampled_ids"]["1"]]
    selected = train.drop_duplicates("id").set_index("id").loc[ids].reset_index()
    selected = selected.dropna(subset=["s2_0_path", "s2_plume_path"])
    records = selected[["id", "plume_id", "s2_0_path", "s2_plume_path"]].to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(one_row, records))
    output = pd.DataFrame(rows)
    summary = {
        "rows": len(output),
        "mask_nonempty": int((output["mask_pixels"] > 0).sum()),
        "contrast_definition": "(mask mean - local dilated-ring mean) / local ring std",
        "swir_nd_definition": "(B11 - B12) / max(B11 + B12, 1)",
        "null": "same mask rolled +73 rows and +91 columns with wraparound",
        "features": {name: signed_summary(output.dropna(subset=[f"{name}_true_contrast", f"{name}_shifted_contrast"]), name) for name in ("swir_nd", "b11", "b12", "visible")},
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
