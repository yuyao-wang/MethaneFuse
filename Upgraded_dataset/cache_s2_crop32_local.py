#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tifffile

import s2_6time_cdse_legacy512_rebuild as pipeline


IMAGE_COLUMNS = list(pipeline.PATH_COLUMNS.values())
ALIAS_COLUMNS = [
    "image_path",
    "s2_path",
    "s2_-7_path",
    "s2_pre_path",
    "s2_pre_pre_path",
]


def local_path(source: str, remote_root: Path, local_root: Path) -> Path:
    return local_root / Path(source).relative_to(remote_root)


def valid_local(path: Path) -> bool:
    if not pipeline.file_ok(path):
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            return tif.series[0].shape == (pipeline.EXPECTED_BANDS, 32, 32)
    except Exception:
        return False


def cache_one(
    source: Path,
    destination: Path,
    source_512_root: Path,
    buffer_size: int,
) -> tuple[str, str]:
    if destination.is_file() and destination.stat().st_size >= 1024:
        return "exists", str(source)
    try:
        pipeline.copy_file_atomic(source, destination, buffer_size)
        if not valid_local(destination):
            raise ValueError(f"invalid cached TIFF: {destination}")
        return "copied", str(source)
    except Exception:
        source_512 = pipeline.repair_crop32_from_512(source, source_512_root, "deflate")
        pipeline.copy_file_atomic(source, destination, buffer_size)
        if not valid_local(destination):
            raise ValueError(f"invalid cached TIFF after repair: {destination}")
        return f"repaired:{source_512}", str(source)


def cache_split(split: str, args: argparse.Namespace) -> pd.DataFrame:
    remote_root = Path(args.remote_root)
    local_root = Path(args.local_root)
    csv_path = remote_root / f"{split}_patches_32.csv"
    frame = pd.read_csv(csv_path, low_memory=False)
    sources = sorted(
        {
            str(value)
            for column in IMAGE_COLUMNS
            for value in frame[column].astype(str)
        }
    )
    tasks = [
        (
            Path(source),
            local_path(source, remote_root, local_root),
            Path(args.source_512_root),
            max(1, int(args.buffer_mb)) * 1024 * 1024,
        )
        for source in sources
    ]
    completed = 0
    repaired = 0
    batch_files = max(1, int(args.batch_files))
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for start in range(0, len(tasks), batch_files):
            futures = [pool.submit(cache_one, *task) for task in tasks[start : start + batch_files]]
            for future in as_completed(futures):
                status, _ = future.result()
                completed += 1
                repaired += int(status.startswith("repaired:"))
                if completed % max(1, int(args.progress_every)) == 0:
                    pipeline.log(
                        f"cache32 {split} {completed}/{len(tasks)} repaired={repaired}"
                    )

    output = frame.copy()
    for column in [*IMAGE_COLUMNS, *ALIAS_COLUMNS]:
        if column in output.columns:
            output[column] = [
                str(local_path(value, remote_root, local_root))
                for value in output[column].astype(str)
            ]
    output_csv = local_root / f"{split}_patches_32_local.csv"
    pipeline.atomic_csv(output, output_csv)
    pipeline.log(
        f"cache32 wrote {output_csv} rows={len(output)} files={len(tasks)} repaired={repaired}"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-root",
        default="/mnt/engg-niulab/yuyao/final_crop/s2_6time_cdse_legacy512_exact_32",
    )
    parser.add_argument(
        "--local-root",
        default="/diniuvol/yuyao/s2_6time_cdse_legacy512_exact_cache/input32",
    )
    parser.add_argument(
        "--source-512-root",
        default="/mnt/engg-niulab/yuyao/preprocessed_512/S2",
    )
    parser.add_argument("--workers", type=int, default=224)
    parser.add_argument("--batch-files", type=int, default=8192)
    parser.add_argument("--buffer-mb", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    local_root = Path(args.local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    train = cache_split("train", args)
    test = cache_split("test", args)
    pipeline.atomic_csv(
        pd.concat([train, test], ignore_index=True),
        local_root / "all_patches_32_local.csv",
    )
    pipeline.log(f"cache32 complete local_root={local_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
