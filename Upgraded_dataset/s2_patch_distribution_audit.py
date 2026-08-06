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


def balanced_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    per_class = maximum // 2
    pieces = []
    for label in (0, 1):
        subset = frame[pd.to_numeric(frame["label"], errors="raise").eq(label)]
        pieces.append(
            subset.sample(n=min(per_class, len(subset)), random_state=seed + label)
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def cached_path(source: str, cache_root: Path | None) -> Path:
    source_path = Path(source)
    if cache_root is None:
        return source_path
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    suffix = source_path.suffix or ".tif"
    return cache_root / digest[:2] / f"{digest}{suffix}"


def read_chw(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {image.shape}: {path}")
    if image.shape[0] not in {10, 12, 13} and image.shape[-1] in {10, 12, 13}:
        image = np.transpose(image, (2, 0, 1))
    if image.shape[0] != 12:
        raise ValueError(f"expected 12 bands, got {image.shape}: {path}")
    return image


def summarize_file(
    cohort: str,
    label: int,
    timepoint: str,
    path: Path,
) -> dict[str, object]:
    image = read_chw(path)
    values = image.astype(np.float64, copy=False)
    valid = values > 0
    counts = valid.sum(axis=(1, 2)).astype(np.int64)
    sums = np.where(valid, values, 0.0).sum(axis=(1, 2))
    squared_sums = np.where(valid, values * values, 0.0).sum(axis=(1, 2))
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    return {
        "cohort": cohort,
        "label": label,
        "timepoint": timepoint,
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "pixels": int(image.shape[1] * image.shape[2]),
        "counts": counts,
        "sums": sums,
        "squared_sums": squared_sums,
        "means": means,
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    counts = np.stack([record["counts"] for record in records]).sum(axis=0)
    sums = np.stack([record["sums"] for record in records]).sum(axis=0)
    squared_sums = np.stack(
        [record["squared_sums"] for record in records]
    ).sum(axis=0)
    total_pixels = sum(int(record["pixels"]) for record in records)
    means = np.stack([record["means"] for record in records])
    pixel_means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    pixel_variances = np.divide(
        squared_sums,
        counts,
        out=np.zeros_like(squared_sums),
        where=counts > 0,
    ) - pixel_means * pixel_means
    pixel_stds = np.sqrt(np.maximum(pixel_variances, 0.0))
    return {
        "files": len(records),
        "dtypes": sorted({str(record["dtype"]) for record in records}),
        "shapes": sorted({tuple(record["shape"]) for record in records}),
        "nonzero_pixel_mean": pixel_means.tolist(),
        "nonzero_pixel_std": pixel_stds.tolist(),
        "zero_fraction": (1.0 - counts / total_pixels).tolist(),
        "per_image_nonzero_mean_q10": np.quantile(means, 0.1, axis=0).tolist(),
        "per_image_nonzero_mean_q50": np.quantile(means, 0.5, axis=0).tolist(),
        "per_image_nonzero_mean_q90": np.quantile(means, 0.9, axis=0).tolist(),
    }


def load_cohort(
    name: str,
    csv_path: str,
    cache_root: str,
    path_columns: tuple[str, ...],
    staged_maximum: int,
    sample_maximum: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(csv_path, low_memory=False)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    frame = balanced_sample(frame, staged_maximum, seed)
    root = Path(cache_root) if cache_root else None
    missing = []
    resolved = frame.copy()
    for path_column in path_columns:
        resolved[path_column] = [
            str(cached_path(str(source), root))
            for source in frame[path_column].astype(str)
        ]
        missing.extend(
            str(path)
            for path in map(Path, resolved[path_column])
            if not path.is_file()
        )
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} cached paths missing; examples={missing[:5]}"
        )
    resolved = balanced_sample(resolved, sample_maximum, seed + 10_000)
    return resolved, {
        "csv": csv_path,
        "cache_root": cache_root,
        "rows": len(resolved),
        "labels": {
            str(key): int(value)
            for key, value in resolved["label"].value_counts().sort_index().items()
        },
    }


def parse_cohort(value: str) -> tuple[str, str, str]:
    parts = value.split("::")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--cohort must be NAME::CSV::CACHE_ROOT; CACHE_ROOT may be empty"
        )
    return parts[0], parts[1], parts[2]


def main(args: argparse.Namespace) -> None:
    path_columns = tuple(
        value.strip() for value in args.path_columns.split(",") if value.strip()
    )
    cohorts = {}
    tasks = []
    for name, csv_path, cache_root in args.cohort:
        frame, metadata = load_cohort(
            name,
            csv_path,
            cache_root,
            path_columns,
            args.staged_maximum,
            args.sample_maximum,
            args.seed,
        )
        cohorts[name] = metadata
        for row in frame.itertuples(index=False):
            label = int(row.label)
            for timepoint, path_column in zip(args.timepoints, path_columns):
                tasks.append(
                    (
                        name,
                        label,
                        timepoint,
                        Path(getattr(row, path_column)),
                    )
                )

    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(summarize_file, *task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(repr(exc))
            if completed % 5000 == 0 or completed == len(futures):
                print(
                    f"[Audit] {completed}/{len(futures)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"{len(failures)} audit reads failed; examples={failures[:5]}"
        )

    grouped = {}
    for cohort in cohorts:
        grouped[cohort] = {}
        for label in (0, 1):
            grouped[cohort][str(label)] = {}
            for timepoint in args.timepoints:
                subset = [
                    record
                    for record in records
                    if record["cohort"] == cohort
                    and record["label"] == label
                    and record["timepoint"] == timepoint
                ]
                grouped[cohort][str(label)][timepoint] = aggregate(subset)

    report = {
        "cohorts": cohorts,
        "path_columns": path_columns,
        "timepoints": args.timepoints,
        "groups": grouped,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cohorts": cohorts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        action="append",
        type=parse_cohort,
        required=True,
    )
    parser.add_argument(
        "--path-columns",
        default="path_t0,path_seasonal,path_year",
    )
    parser.add_argument(
        "--timepoints",
        nargs="+",
        default=["t0", "seasonal", "year"],
    )
    parser.add_argument("--staged-maximum", type=int, default=10000)
    parser.add_argument("--sample-maximum", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
