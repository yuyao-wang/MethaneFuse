#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = (
    ("t0", "path_t0", "s2_0_std_512"),
    ("seasonal", "path_seasonal", "s2_-90_std_512"),
    ("year", "path_year", "s2_-360_std_512"),
)


def event_key(
    frame: pd.DataFrame,
    latitude: str,
    longitude: str,
    timestamp: str,
) -> pd.Series:
    return (
        pd.to_numeric(frame[latitude], errors="raise").round(6).astype(str)
        + "|"
        + pd.to_numeric(frame[longitude], errors="raise").round(6).astype(str)
        + "|"
        + pd.to_datetime(frame[timestamp], utc=True, errors="raise").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    )


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D TIFF, got {array.shape}")
    if array.shape[0] == 12:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] == 12:
        return np.moveaxis(array, -1, 0).astype(np.float32, copy=False)
    raise ValueError(f"Cannot identify channel axis in {array.shape}")


@lru_cache(maxsize=512)
def read_chw(path: str) -> np.ndarray:
    return to_chw(tifffile.imread(path))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    if left_flat.std() == 0 or right_flat.std() == 0:
        return float(left_flat.std() == 0 and right_flat.std() == 0)
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def match_one(
    old_row: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    old_t0 = read_chw(str(old_row["path_t0"]))
    best: dict[str, Any] | None = None
    for source_row in source_rows:
        source_t0 = read_chw(str(source_row["s2_0_std_512"]))
        result = cv2.matchTemplate(
            source_t0[11],
            old_t0[11],
            cv2.TM_CCOEFF_NORMED,
        )
        _, score, _, location = cv2.minMaxLoc(result)
        candidate = {
            "score": float(score),
            "location": location,
            "source_row": source_row,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        raise RuntimeError("No source candidates")

    x, y = best["location"]
    source_row = best["source_row"]
    timepoint_metrics: dict[str, Any] = {}
    for name, old_column, source_column in TIMEPOINTS:
        old_patch = read_chw(str(old_row[old_column]))
        source = read_chw(str(source_row[source_column]))
        source_patch = source[:, y : y + 32, x : x + 32]
        if source_patch.shape != old_patch.shape:
            raise ValueError(
                f"Bad matched crop {source_patch.shape} at {(x, y)} for {source_column}"
            )
        difference = old_patch.astype(np.float64) - source_patch.astype(np.float64)
        timepoint_metrics[name] = {
            "correlation": correlation(old_patch, source_patch),
            "mae": float(np.abs(difference).mean()),
            "band_bias": difference.mean(axis=(1, 2)).tolist(),
            "band_mae": np.abs(difference).mean(axis=(1, 2)).tolist(),
            "exact": bool(np.array_equal(old_patch, source_patch)),
        }
    return {
        "old_path": old_row["path_t0"],
        "label": int(old_row["label"]),
        "event_key": old_row["_event_key"],
        "plume_id": source_row["plume_id"],
        "match_score_band11": best["score"],
        "match_x": int(x),
        "match_y": int(y),
        "timepoints": timepoint_metrics,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "rows": len(results),
        "match_score_band11": {},
        "timepoints": {},
    }
    scores = np.asarray(
        [row["match_score_band11"] for row in results],
        dtype=np.float64,
    )
    output["match_score_band11"] = {
        "median": float(np.median(scores)),
        "p10": float(np.quantile(scores, 0.10)),
        "p90": float(np.quantile(scores, 0.90)),
        "at_least_0_99": int((scores >= 0.99).sum()),
        "at_least_0_90": int((scores >= 0.90).sum()),
    }
    for name, _, _ in TIMEPOINTS:
        correlations = np.asarray(
            [row["timepoints"][name]["correlation"] for row in results],
            dtype=np.float64,
        )
        maes = np.asarray(
            [row["timepoints"][name]["mae"] for row in results],
            dtype=np.float64,
        )
        band_bias = np.asarray(
            [row["timepoints"][name]["band_bias"] for row in results],
            dtype=np.float64,
        )
        band_mae = np.asarray(
            [row["timepoints"][name]["band_mae"] for row in results],
            dtype=np.float64,
        )
        output["timepoints"][name] = {
            "exact": int(
                sum(row["timepoints"][name]["exact"] for row in results)
            ),
            "correlation_median": float(np.median(correlations)),
            "correlation_p10": float(np.quantile(correlations, 0.10)),
            "mae_median": float(np.median(maes)),
            "band_bias_median": np.median(band_bias, axis=0).tolist(),
            "band_mae_median": np.median(band_mae, axis=0).tolist(),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    old = pd.read_csv(args.old_csv, low_memory=False)
    source = pd.read_csv(args.source_csv, low_memory=False)
    old["_event_key"] = event_key(
        old,
        "latitude",
        "longitude",
        "datetime",
    )
    source["_event_key"] = event_key(
        source,
        "plume_latitude",
        "plume_longitude",
        "event_time",
    )
    source_groups = {
        key: group.to_dict("records")
        for key, group in source.groupby("_event_key", sort=False)
    }
    old = old[old["_event_key"].isin(source_groups)].copy()
    if 0 < args.samples < len(old):
        per_label = max(1, args.samples // 2)
        pieces = []
        for label in (0, 1):
            subset = old[old["label"].eq(label)]
            pieces.append(
                subset.sample(
                    n=min(per_label, len(subset)),
                    random_state=args.seed + label,
                )
            )
        old = pd.concat(pieces, ignore_index=True)

    records = old.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(
            pool.map(
                lambda row: match_one(
                    row,
                    source_groups[row["_event_key"]],
                ),
                records,
            )
        )
    report = {
        "old_csv": args.old_csv,
        "source_csv": args.source_csv,
        "summary": aggregate(results),
        "examples": results[:20],
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
