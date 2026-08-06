#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = {
    "t0": ("s2_0_std_512", "t0_input_kind"),
    "prev1": ("s2_-7_std_512", "prev1_input_kind"),
    "prev2": ("s2_prev2_std_512", "prev2_input_kind"),
    "prev3": ("s2_prev3_std_512", "prev3_input_kind"),
    "seasonal": ("s2_-90_std_512", "seasonal_input_kind"),
    "year": ("s2_-360_std_512", "year_input_kind"),
}
REPAIR_KINDS = {
    "point_exact_v3_harmonized",
    "local_safe_point_recrop_v4",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.shape == (12, 512, 512):
        return array
    if array.shape == (512, 512, 12):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"unexpected shape={array.shape}")


def audit_image(path_text: str) -> str:
    try:
        image = to_chw(tifffile.imread(path_text))
        if not np.any(image):
            return "all_zero_image"
        empty = [band for band in (7, 8, 9) if not np.any(image[band])]
        if empty:
            return f"empty_bands={empty}"
        if np.array_equal(image[7], image[8]):
            return "B08_equals_B8A"
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--skip-targets-json", default="")
    parser.add_argument("--sample-images", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, low_memory=False)
    skipped: set[tuple[str, str]] = set()
    if args.skip_targets_json:
        skipped = {
            (clean(item[0]), clean(item[1]))
            for item in json.loads(Path(args.skip_targets_json).read_text())
            if isinstance(item, list) and len(item) >= 2
        }

    all_paths: set[str] = set()
    candidates: list[tuple[str, str, str]] = []
    for row in frame.to_dict("records"):
        plume_id = clean(row["plume_id"])
        for timepoint, (path_column, kind_column) in TIMEPOINTS.items():
            path = clean(row[path_column])
            all_paths.add(path)
            if clean(row[kind_column]) in REPAIR_KINDS:
                candidates.append((plume_id, timepoint, path))

    marker_missing: list[str] = []
    bad_metadata: list[str] = []
    canonical_markers = 0
    explicit_skips = 0
    for plume_id, timepoint, path_text in candidates:
        if (plume_id, timepoint) in skipped:
            explicit_skips += 1
            continue
        sidecar = Path(path_text).with_name(Path(path_text).name + ".georef.json")
        try:
            metadata = json.loads(sidecar.read_text())
        except Exception as exc:
            marker_missing.append(f"{path_text}: {type(exc).__name__}: {exc}")
            continue
        if not metadata.get("canonical_s2_band_fill_complete"):
            marker_missing.append(path_text)
            continue
        canonical_markers += 1
        resampling = tuple(
            metadata.get(key)
            for key in ("b08_resampling", "b8a_resampling", "b09_resampling")
        )
        if resampling != ("nearest", "nearest", "nearest"):
            bad_metadata.append(f"{path_text}: resampling={resampling}")

    ordered_paths = sorted(all_paths)
    if 0 < args.sample_images < len(ordered_paths):
        sampled_paths = random.Random(args.seed).sample(
            ordered_paths,
            args.sample_images,
        )
    else:
        sampled_paths = ordered_paths
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        errors = list(pool.map(audit_image, sampled_paths))
    bad_images = [
        {"path": path, "error": error}
        for path, error in zip(sampled_paths, errors)
        if error and error != "all_zero_image"
    ]
    all_zero_images = sum(error == "all_zero_image" for error in errors)
    report = {
        "rows": len(frame),
        "unique_512_paths": len(all_paths),
        "candidate_targets": len(candidates),
        "canonical_markers": canonical_markers,
        "explicit_skips": explicit_skips,
        "marker_missing": len(marker_missing),
        "marker_missing_examples": marker_missing[:20],
        "bad_metadata": len(bad_metadata),
        "bad_metadata_examples": bad_metadata[:20],
        "sampled_images": len(sampled_paths),
        "all_zero_sampled_images": all_zero_images,
        "bad_sampled_images": len(bad_images),
        "bad_sampled_image_examples": bad_images[:20],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    print(text, flush=True)
    return int(
        bool(
            marker_missing
            or bad_metadata
            or bad_images
            or canonical_markers + explicit_skips != len(candidates)
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
