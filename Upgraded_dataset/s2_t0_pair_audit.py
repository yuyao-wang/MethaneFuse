#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


MODEL_BANDS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)


def read_chw(path: str) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.shape[-1] == 12 and image.shape[0] != 12:
        image = np.transpose(image, (2, 0, 1))
    if image.shape != (12, 32, 32):
        raise ValueError(f"unexpected image shape {image.shape}: {path}")
    return image.astype(np.float64, copy=False)


def compare_pair(old_path: str, new_path: str) -> dict[str, object]:
    old = read_chw(old_path)
    new = read_chw(new_path)
    correlations = []
    slopes = []
    intercepts = []
    mean_absolute_errors = []
    old_zero_fractions = []
    new_zero_fractions = []
    old_means = []
    new_means = []
    for band in MODEL_BANDS:
        old_band = old[band].ravel()
        new_band = new[band].ravel()
        valid = (old_band > 0) & (new_band > 0)
        if valid.sum() >= 16:
            old_valid = old_band[valid]
            new_valid = new_band[valid]
            correlation = np.corrcoef(old_valid, new_valid)[0, 1]
            slope, intercept = np.polyfit(new_valid, old_valid, 1)
            mean_absolute_error = np.abs(old_valid - new_valid).mean()
        else:
            correlation = np.nan
            slope = np.nan
            intercept = np.nan
            mean_absolute_error = np.nan
        correlations.append(correlation)
        slopes.append(slope)
        intercepts.append(intercept)
        mean_absolute_errors.append(mean_absolute_error)
        old_zero_fractions.append((old_band == 0).mean())
        new_zero_fractions.append((new_band == 0).mean())
        old_means.append(old_band[old_band > 0].mean() if np.any(old_band > 0) else 0)
        new_means.append(new_band[new_band > 0].mean() if np.any(new_band > 0) else 0)
    return {
        "correlations": correlations,
        "slopes": slopes,
        "intercepts": intercepts,
        "mean_absolute_errors": mean_absolute_errors,
        "old_zero_fractions": old_zero_fractions,
        "new_zero_fractions": new_zero_fractions,
        "old_means": old_means,
        "new_means": new_means,
    }


def resolve_cached_path(source: str, cache_root: str) -> Path:
    source_path = Path(source)
    if not cache_root:
        return source_path
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    suffix = source_path.suffix or ".tif"
    root = Path(cache_root)
    candidates = [
        root / "train" / digest[:2] / f"{digest}{suffix}",
        root / "test" / digest[:2] / f"{digest}{suffix}",
        root / digest[:2] / f"{digest}{suffix}",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def median_array(records: list[dict[str, object]], key: str) -> list[float]:
    return np.nanmedian(np.asarray([record[key] for record in records]), axis=0).tolist()


def main(args: argparse.Namespace) -> None:
    original = pd.read_csv(args.original_csv, low_memory=False)
    corrected = pd.read_csv(args.corrected_csv, low_memory=False)
    original_paths = original.set_index(args.key_column)["path_t0"]
    pairs = []
    for row in corrected.itertuples(index=False):
        key = getattr(row, args.key_column)
        if key not in original_paths.index:
            continue
        new_path = str(original_paths.loc[key])
        cached_new_path = resolve_cached_path(new_path, args.new_cache_root)
        if not Path(row.path_t0).is_file() or not cached_new_path.is_file():
            continue
        pairs.append((str(row.path_t0), str(cached_new_path)))
    pairs_frame = pd.DataFrame(pairs, columns=["old_path", "new_path"]).drop_duplicates()
    if args.max_pairs > 0 and len(pairs_frame) > args.max_pairs:
        pairs_frame = pairs_frame.sample(n=args.max_pairs, random_state=args.seed)

    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(compare_pair, row.old_path, row.new_path)
            for row in pairs_frame.itertuples(index=False)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(repr(exc))
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"[Pairs] {completed}/{len(futures)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"pair failures={len(failures)} examples={failures[:5]}")

    report = {
        "original_csv": args.original_csv,
        "corrected_csv": args.corrected_csv,
        "new_cache_root": args.new_cache_root,
        "pairs": len(records),
        "model_bands": MODEL_BANDS,
        "median_correlation": median_array(records, "correlations"),
        "median_old_from_new_slope": median_array(records, "slopes"),
        "median_old_from_new_intercept": median_array(records, "intercepts"),
        "median_mean_absolute_error": median_array(
            records, "mean_absolute_errors"
        ),
        "median_old_zero_fraction": median_array(records, "old_zero_fractions"),
        "median_new_zero_fraction": median_array(records, "new_zero_fractions"),
        "median_old_nonzero_mean": median_array(records, "old_means"),
        "median_new_nonzero_mean": median_array(records, "new_means"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--corrected-csv", required=True)
    parser.add_argument("--key-column", default="path")
    parser.add_argument("--new-cache-root", default="")
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
