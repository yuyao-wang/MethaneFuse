#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


MODEL_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def to_chw(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {array.shape}: {path}")
    if array.shape[0] == 12:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] == 12:
        return np.moveaxis(array, -1, 0).astype(np.float32, copy=False)
    raise ValueError(f"expected 12 bands, got {array.shape}: {path}")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if int(valid.sum()) < 100:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def compare(row: dict[str, Any], legacy_root: Path) -> dict[str, Any]:
    plume_id = clean(row["plume_id"])
    current_path = Path(clean(row["t0_v8_source_path"]))
    legacy_path = legacy_root / plume_id / "s2_std_512.tif"
    current = to_chw(current_path)
    legacy = to_chw(legacy_path)
    current_model = current[list(MODEL_BANDS)]
    legacy_model = legacy[list(MODEL_BANDS)]
    valid = (current_model > 0) & (legacy_model > 0)
    delta = current_model[valid] - legacy_model[valid]
    central = np.s_[..., 64:448, 64:448]
    return {
        "plume_id": plume_id,
        "current_path": str(current_path),
        "legacy_path": str(legacy_path),
        "current_zero_fraction": float(np.mean(current_model == 0)),
        "legacy_zero_fraction": float(np.mean(legacy_model == 0)),
        "current_mean_nonzero": float(current_model[current_model > 0].mean()),
        "legacy_mean_nonzero": float(legacy_model[legacy_model > 0].mean()),
        "same_pixel_correlation": correlation(current_model, legacy_model),
        "central_correlation": correlation(
            current_model[central],
            legacy_model[central],
        ),
        "median_current_minus_legacy": float(np.median(delta)),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)])
    return {
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument(
        "--legacy-root",
        default=(
            "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/"
            "plume_raw_s2_90360_fixed_512"
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.concat(
        [
            pd.read_csv(args.train_csv, low_memory=False),
            pd.read_csv(args.test_csv, low_memory=False),
        ],
        ignore_index=True,
    )
    legacy_root = Path(args.legacy_root)
    selected = [
        row
        for row in frame.to_dict("records")
        if "S2_point_center_exact_v3" in clean(row.get("t0_v8_source_path"))
        and (legacy_root / clean(row.get("plume_id")) / "s2_std_512.tif").is_file()
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(lambda row: compare(row, legacy_root), selected))

    metric_names = [
        "current_zero_fraction",
        "legacy_zero_fraction",
        "current_mean_nonzero",
        "legacy_mean_nonzero",
        "same_pixel_correlation",
        "central_correlation",
        "median_current_minus_legacy",
        "mean_absolute_error",
    ]
    report = {
        "pairs": len(records),
        "model_bands": list(MODEL_BANDS),
        "summary": {
            metric: quantiles([float(record[metric]) for record in records])
            for metric in metric_names
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": len(records), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
