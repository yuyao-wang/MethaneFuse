#!/usr/bin/env python3
"""Create dataset_part_XXX.tar.gz archives from hf_dataset_release/file_manifest.csv.

The script streams source files into tar.gz archives using the target_path as arcname,
so files are renamed in the archive without first copying the dataset locally.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import pandas as pd


def iter_manifest(path: Path, chunksize: int = 100_000):
    seen_targets: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            source = str(row.source_path)
            target = str(row.target_path)
            if not source or not target or target in seen_targets:
                continue
            seen_targets.add(target)
            yield source, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="hf_dataset_release/file_manifest.csv")
    parser.add_argument("--out-dir", default="hf_dataset_parts")
    parser.add_argument("--max-uncompressed-gb", type=float, default=8.0)
    parser.add_argument("--start-part", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    max_bytes = int(args.max_uncompressed_gb * 1024**3)
    out_dir.mkdir(parents=True, exist_ok=True)

    part = args.start_part
    current_size = 0
    current_count = 0
    total_size = 0
    total_count = 0
    missing = []
    tar = None

    def open_tar(part_no: int):
        out_path = out_dir / f"dataset_part_{part_no:03d}.tar.gz"
        return tarfile.open(out_path, "w:gz"), out_path

    tar_path = None
    if not args.dry_run:
        tar, tar_path = open_tar(part)

    part_stats = []
    for source, target in iter_manifest(manifest):
        src = Path(source)
        try:
            size = src.stat().st_size
        except FileNotFoundError:
            missing.append({"source_path": source, "target_path": target})
            continue
        if current_count and current_size + size > max_bytes:
            if tar is not None:
                tar.close()
            part_stats.append({"part": part, "path": str(tar_path), "files": current_count, "uncompressed_bytes": current_size})
            part += 1
            current_size = 0
            current_count = 0
            if not args.dry_run:
                tar, tar_path = open_tar(part)
        if not args.dry_run:
            tar.add(src, arcname=target, recursive=False)
        current_size += size
        total_size += size
        current_count += 1
        total_count += 1

    if tar is not None:
        tar.close()
    if current_count:
        part_stats.append({"part": part, "path": str(tar_path), "files": current_count, "uncompressed_bytes": current_size})

    summary = {
        "dry_run": args.dry_run,
        "total_files": total_count,
        "total_uncompressed_bytes": total_size,
        "total_uncompressed_gb": total_size / 1024**3,
        "missing_files": len(missing),
        "parts": part_stats,
    }
    (out_dir / "packaging_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if missing:
        pd.DataFrame(missing).to_csv(out_dir / "missing_files.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
