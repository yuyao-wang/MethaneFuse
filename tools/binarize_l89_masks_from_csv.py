#!/usr/bin/env python3
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

NULL_LIKE = {"", "none", "nan", "null"}


def _clean(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in NULL_LIKE:
        return ""
    return s


def _collect_unique_l89_paths(csv_files: Sequence[str]) -> List[str]:
    out: Set[str] = set()
    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "l89_plume_path" not in (reader.fieldnames or []):
                raise ValueError(f"{csv_path} has no 'l89_plume_path' column")
            for row in reader:
                p = _clean(row.get("l89_plume_path"))
                if p:
                    out.add(p)
    return sorted(out)


def _to_binary(arr: np.ndarray, threshold: float) -> np.ndarray:
    return (arr > threshold).astype(arr.dtype, copy=False)


def _process_one(args: Tuple[str, float, bool]) -> Tuple[str, str, str]:
    path, threshold, dry_run = args
    if not os.path.exists(path):
        return ("missing", path, "file_not_found")

    tmp_path = f"{path}.tmp_bin"
    try:
        with rasterio.open(path) as ds:
            arr = ds.read()
            profile = ds.profile.copy()

        bin_arr = _to_binary(arr, threshold=threshold)
        changed = not np.array_equal(arr, bin_arr)
        if not changed:
            return ("unchanged", path, "")
        if dry_run:
            return ("would_change", path, "")

        # Write to temp and atomically replace the original file.
        with rasterio.open(tmp_path, "w", **profile) as out_ds:
            out_ds.write(bin_arr)

        os.replace(tmp_path, path)
        return ("changed", path, "")
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return ("error", path, str(e))


def _batched_tasks(paths: Iterable[str], threshold: float, dry_run: bool):
    for p in paths:
        yield (p, threshold, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize all L89 plume masks referenced by one or more CSV files."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.0, help="Mask binarization threshold: value > threshold -> 1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--max_files", type=int, default=0, help="If > 0, only process first N unique files (for quick check).")
    args = parser.parse_args()

    csv_files = [args.train_csv, args.test_csv]
    paths = _collect_unique_l89_paths(csv_files)
    if args.max_files > 0:
        paths = paths[: args.max_files]
    total = len(paths)
    if total == 0:
        raise SystemExit("No valid l89_plume_path found in CSVs.")

    print(f"Collected {total} unique L89 mask files.")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'IN-PLACE WRITE'} | threshold={args.threshold} | workers={args.workers}")

    counts = {
        "changed": 0,
        "would_change": 0,
        "unchanged": 0,
        "missing": 0,
        "error": 0,
    }
    errors: List[Tuple[str, str]] = []

    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_process_one, t) for t in _batched_tasks(paths, args.threshold, args.dry_run)]
        for fut in as_completed(futs):
            status, path, msg = fut.result()
            counts[status] += 1
            done += 1
            if status == "error":
                errors.append((path, msg))
            if done % args.log_every == 0 or done == total:
                print(
                    f"[{done}/{total}] changed={counts['changed']} would_change={counts['would_change']} "
                    f"unchanged={counts['unchanged']} missing={counts['missing']} error={counts['error']}"
                )

    print("Done.")
    print(counts)
    if errors:
        print("First errors:")
        for p, m in errors[:20]:
            print(f"- {p} :: {m}")


if __name__ == "__main__":
    main()
