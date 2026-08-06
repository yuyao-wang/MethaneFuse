#!/usr/bin/env python3
"""Extract compact image-quality and band statistics from cached L89 TIFFs."""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


PATH_COLUMNS = (
    "path_t0",
    "path_prev1",
    "path_prev2",
    "path_prev3",
    "path_seasonal",
    "path_year",
)


def cached_path(original: str, cache_dir: str) -> Path:
    normalized = os.path.abspath(original)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return Path(cache_dir) / digest[:2] / f"{digest}{Path(original).suffix}"


def summarize_image(path: str) -> dict[str, float | int | str]:
    try:
        image = tifffile.imread(path)
        if image.ndim != 3 or image.shape[0] < 7:
            raise ValueError(f"unexpected shape {image.shape}")
        bands = image[:7].astype(np.float32, copy=False)
        finite = np.isfinite(bands)
        valid = finite & (bands != 0)
        result: dict[str, float | int | str] = {
            "read_ok": 1,
            "zero_fraction": float((bands == 0).mean()),
            "nonfinite_fraction": float((~finite).mean()),
            "extreme_fraction": float(((bands < 1) | (bands > 30000)).mean()),
        }
        for band_index in range(7):
            values = bands[band_index][valid[band_index]]
            prefix = f"b{band_index + 1}"
            if values.size:
                result[f"{prefix}_mean"] = float(values.mean())
                result[f"{prefix}_std"] = float(values.std())
                result[f"{prefix}_p05"] = float(np.quantile(values, 0.05))
                result[f"{prefix}_p95"] = float(np.quantile(values, 0.95))
            else:
                for suffix in ("mean", "std", "p05", "p95"):
                    result[f"{prefix}_{suffix}"] = float("nan")
        return result
    except Exception as exc:
        return {"read_ok": 0, "read_error": f"{type(exc).__name__}: {exc}"}


def summarize_row(task: tuple[int, tuple[str, ...]]) -> dict:
    row_index, paths = task
    result: dict[str, float | int | str] = {"row_index": row_index}
    for timepoint, path in zip(PATH_COLUMNS, paths):
        stats = summarize_image(path)
        result.update({f"{timepoint}_{key}": value for key, value in stats.items()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cache_dir", default="/diniuvol/yuyao/l89_temporal_cache")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--progress_every", type=int, default=500)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, usecols=["id", *PATH_COLUMNS], low_memory=False)
    tasks = []
    missing = 0
    for row_index, row in frame.iterrows():
        paths = []
        for column in PATH_COLUMNS:
            local_path = cached_path(str(row[column]), args.cache_dir)
            if not local_path.is_file():
                missing += 1
            paths.append(str(local_path))
        tasks.append((int(row_index), tuple(paths)))

    print(
        f"[Scan] rows={len(frame)} files={len(frame) * len(PATH_COLUMNS)} "
        f"missing_cached_files={missing}",
        flush=True,
    )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for completed, result in enumerate(
            pool.map(summarize_row, tasks, chunksize=args.chunksize), 1
        ):
            rows.append(result)
            if completed % args.progress_every == 0:
                print(f"[Scan] {completed}/{len(tasks)} rows", flush=True)

    stats = pd.DataFrame(rows).sort_values("row_index")
    output = pd.concat(
        [frame[["id"]].reset_index(drop=True), stats.reset_index(drop=True)], axis=1
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    failures = sum(
        int((output[f"{column}_read_ok"] != 1).sum())
        for column in PATH_COLUMNS
        if f"{column}_read_ok" in output
    )
    print(f"[Scan] wrote {args.output_csv}; read_failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
