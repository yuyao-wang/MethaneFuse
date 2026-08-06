#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from affine import Affine
from rasterio.warp import transform as transform_coordinates


TIMEPOINTS = {
    "t0": ("t0_raw_path", "path_t0"),
    "prev1": ("prev1_raw_path", "path_prev1"),
    "prev2": ("prev2_raw_path", "path_prev2"),
    "prev3": ("prev3_raw_path", "path_prev3"),
    "seasonal": ("seasonal_raw_path", "path_seasonal"),
    "year": ("year_raw_path", "path_year"),
}
VALID_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def to_chw(path: str) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {array.shape}: {path}")
    if array.shape[0] == 12:
        return array
    if array.shape[-1] == 12:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"expected 12 bands, got {array.shape}: {path}")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if int(valid.sum()) < 64:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def inspect_source(task: tuple[str, str]) -> dict[str, Any]:
    timepoint, path = task
    try:
        array = to_chw(path)
        center = array[list(VALID_BANDS), 240:272, 240:272]
        center_zero_fractions = np.mean(center == 0, axis=(1, 2))
        return {
            "timepoint": timepoint,
            "path": path,
            "ok": array.shape == (12, 512, 512),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "muted_nonzero": int(np.count_nonzero(array[[8, 9]])),
            "valid_nonzero": int(np.count_nonzero(array[list(VALID_BANDS)])),
            "center_max_valid_band_zero_fraction": float(
                np.max(center_zero_fractions)
            ),
        }
    except Exception as error:
        return {
            "timepoint": timepoint,
            "path": path,
            "ok": False,
            "error": f"{type(error).__name__}:{error}",
        }


def inspect_center(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["t0_raw_path"]))
    sidecar = path.with_name(path.name + ".georef.json")
    try:
        metadata = json.loads(sidecar.read_text())
        x, y = transform_coordinates(
            "EPSG:4326",
            metadata["crs_wkt"],
            [float(row["plume_longitude"])],
            [float(row["plume_latitude"])],
        )
        inverse = ~Affine(*metadata["transform"][:6])
        column, line = inverse * (x[0], y[0])
        return {
            "plume_id": str(row["plume_id"]),
            "ok": True,
            "column": float(column),
            "line": float(line),
            "center_error_pixels": float(
                max(abs(column - 256.0), abs(line - 256.0))
            ),
            "source_dn_add": metadata.get("source_dn_add"),
            "source_kind": metadata.get("source_kind"),
            "source_product_exact": metadata.get("source_product_exact"),
        }
    except Exception as error:
        return {
            "plume_id": str(row["plume_id"]),
            "ok": False,
            "error": f"{type(error).__name__}:{error}",
        }


def compare_patch(
    task: tuple[dict[str, Any], dict[str, Any], list[str]],
) -> dict[str, Any]:
    patch_row, source_row, timepoints = task
    crop_x = int(patch_row["crop_x"])
    crop_y = int(patch_row["crop_y"])
    result: dict[str, Any] = {
        "plume_id": str(patch_row["plume_id"]),
        "crop_x": crop_x,
        "crop_y": crop_y,
        "timepoints": {},
    }
    for timepoint in timepoints:
        source_column, patch_column = TIMEPOINTS[timepoint]
        source = to_chw(str(source_row[source_column])).astype(
            np.float32, copy=False
        )
        patch = to_chw(str(patch_row[patch_column])).astype(
            np.float32, copy=False
        )
        expected = source[:, crop_y : crop_y + 32, crop_x : crop_x + 32]
        band_correlations = [
            correlation(patch[band], expected[band]) for band in VALID_BANDS
        ]
        band_mae = [
            float(np.mean(np.abs(patch[band] - expected[band])))
            for band in VALID_BANDS
        ]
        result["timepoints"][timepoint] = {
            "band_correlations": band_correlations,
            "band_mae": band_mae,
            "band11_correlation": band_correlations[-1],
            "median_valid_band_correlation": float(
                np.nanmedian(band_correlations)
            ),
            "median_valid_band_mae": float(np.median(band_mae)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--patch-csv", required=True)
    parser.add_argument("--timepoints", default="t0,seasonal,year")
    parser.add_argument("--sample-plumes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    timepoints = [
        value.strip() for value in args.timepoints.split(",") if value.strip()
    ]
    invalid = sorted(set(timepoints) - set(TIMEPOINTS))
    if invalid:
        raise ValueError(f"unsupported timepoints: {invalid}")

    source = pd.read_csv(args.source_csv, low_memory=False)
    patches = pd.read_csv(args.patch_csv, low_memory=False)
    source_lookup = {
        str(row["plume_id"]): row for row in source.to_dict("records")
    }

    source_tasks = [
        (timepoint, str(row[TIMEPOINTS[timepoint][0]]))
        for row in source.to_dict("records")
        for timepoint in timepoints
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        source_checks = list(executor.map(inspect_source, source_tasks))
        center_checks = (
            list(executor.map(inspect_center, source.to_dict("records")))
            if "t0" in timepoints
            else []
        )

    sample = patches[pd.to_numeric(patches["label"]).eq(1)].copy()
    sample = sample.sort_values(["plume_id", "crop_index"], kind="stable")
    sample = sample.groupby("plume_id", sort=False).head(1)
    sample = sample[sample["plume_id"].astype(str).isin(source_lookup)]
    sample = sample.sample(
        n=min(args.sample_plumes, len(sample)),
        random_state=20260724,
    )
    comparison_tasks = [
        (row, source_lookup[str(row["plume_id"])], timepoints)
        for row in sample.to_dict("records")
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        comparisons = list(executor.map(compare_patch, comparison_tasks))

    source_summary: dict[str, Any] = {}
    comparison_summary: dict[str, Any] = {}
    for timepoint in timepoints:
        checks = [
            record for record in source_checks if record["timepoint"] == timepoint
        ]
        source_summary[timepoint] = {
            "files": len(checks),
            "valid_files": int(sum(bool(record["ok"]) for record in checks)),
            "muted_nonzero": int(
                sum(int(record.get("muted_nonzero", 0)) for record in checks)
            ),
            "files_without_valid_pixels": int(
                sum(int(record.get("valid_nonzero", 0)) == 0 for record in checks)
            ),
            "files_with_invalid_center": int(
                sum(
                    float(
                        record.get(
                            "center_max_valid_band_zero_fraction",
                            1.0,
                        )
                    )
                    >= 0.20
                    for record in checks
                )
            ),
            "dtypes": sorted(
                {str(record.get("dtype")) for record in checks if record["ok"]}
            ),
        }
        values = [
            record["timepoints"][timepoint] for record in comparisons
        ]
        comparison_summary[timepoint] = {
            "samples": len(values),
            "median_band11_correlation": float(
                np.nanmedian([value["band11_correlation"] for value in values])
            ),
            "median_valid_band_correlation": float(
                np.nanmedian(
                    [value["median_valid_band_correlation"] for value in values]
                )
            ),
            "median_valid_band_mae": float(
                np.median([value["median_valid_band_mae"] for value in values])
            ),
        }

    center_errors = [
        record["center_error_pixels"]
        for record in center_checks
        if record["ok"]
    ]
    center_summary = {
        "files": len(center_checks),
        "valid_sidecars": int(sum(bool(record["ok"]) for record in center_checks)),
        "maximum_center_error_pixels": (
            float(max(center_errors)) if center_errors else None
        ),
        "median_center_error_pixels": float(np.median(center_errors))
        if center_errors
        else None,
        "source_dn_add_counts": pd.Series(
            [
                str(record.get("source_dn_add"))
                for record in center_checks
                if record["ok"]
            ]
        )
        .value_counts()
        .to_dict(),
        "source_kind_counts": pd.Series(
            [
                str(record.get("source_kind"))
                for record in center_checks
                if record["ok"]
            ]
        )
        .value_counts()
        .to_dict(),
    }

    failures: list[str] = []
    for timepoint, summary in source_summary.items():
        if summary["valid_files"] != summary["files"]:
            failures.append(f"{timepoint}: invalid or unreadable files")
        if summary["muted_nonzero"] != 0:
            failures.append(f"{timepoint}: bands 8/9 are not fully muted")
        if summary["files_without_valid_pixels"] != 0:
            failures.append(f"{timepoint}: files without valid-band pixels")
        if summary["files_with_invalid_center"] != 0:
            failures.append(
                f"{timepoint}: files fail the old 20% center zero threshold"
            )
    if "t0" in timepoints:
        if center_summary["valid_sidecars"] != center_summary["files"]:
            failures.append("missing or invalid t0 georeference sidecars")
        if center_summary["maximum_center_error_pixels"] > 1.0:
            failures.append("plume point is not centered in every t0 crop")
    for timepoint, summary in comparison_summary.items():
        if summary["median_band11_correlation"] < 0.85:
            failures.append(f"{timepoint}: local patch alignment correlation too low")

    report = {
        "passed": not failures,
        "failures": failures,
        "source_rows": len(source),
        "source_summary": source_summary,
        "center_summary": center_summary,
        "comparison_summary": comparison_summary,
        "comparison_details": comparisons,
        "failed_source_details": [
            record for record in source_checks if not record["ok"]
        ][:100],
        "failed_center_details": [
            record for record in center_checks if not record["ok"]
        ][:100],
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "failures": failures,
                "source_summary": source_summary,
                "center_summary": center_summary,
                "comparison_summary": comparison_summary,
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
