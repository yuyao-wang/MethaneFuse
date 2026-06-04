#!/usr/bin/env python3
"""Rewrite query manifest file paths after extracting an archive on a compute node."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rewrite_csv(path: Path, old_root: str, new_root: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    old_prefix = old_root.rstrip("/") + "/"
    new_prefix = new_root.rstrip("/") + "/"

    with path.open(newline="") as src, tmp_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            for key, value in row.items():
                if isinstance(value, str) and value.startswith(old_prefix):
                    row[key] = new_prefix + value[len(old_prefix) :]
            writer.writerow(row)

    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--old-root", required=True)
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=["manifest_time_train.csv", "manifest_time_test.csv"],
    )
    args = parser.parse_args()

    for name in args.manifests:
        path = args.data_root / name
        if not path.exists():
            raise FileNotFoundError(path)
        rewrite_csv(path, args.old_root, str(args.data_root))
        print(f"rewrote {path}")


if __name__ == "__main__":
    main()
