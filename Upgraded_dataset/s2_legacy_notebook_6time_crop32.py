#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = {
    "t0": ("s2_0_std_512", "s2_0.tif", "t0_image_time"),
    "prev1": ("s2_-7_std_512", "s2_prev1.tif", "prev1_image_time"),
    "prev2": ("s2_prev2_std_512", "s2_prev2.tif", "prev2_image_time"),
    "prev3": ("s2_prev3_std_512", "s2_prev3.tif", "prev3_image_time"),
    "seasonal": (
        "s2_-90_std_512",
        "s2_seasonal.tif",
        "seasonal_image_time",
    ),
    "year": ("s2_-360_std_512", "s2_year.tif", "year_image_time"),
}
PATCH_SIZE = 32
BAND_INDEX = 11
ZERO_RATIO_THRESHOLD = 0.20


def stable_seed(seed: int, plume_id: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{plume_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def to_chw(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[0] in (12, 13):
        pass
    elif array.ndim == 3 and array.shape[-1] in (12, 13):
        array = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"unsupported TIFF shape {array.shape}: {path}")
    if array.shape != (12, 512, 512):
        raise ValueError(f"expected 12x512x512, got {array.shape}: {path}")
    return array.astype(np.float32, copy=False)


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(tifffile.imread(path))
    mask = np.squeeze(mask)
    if mask.shape != (512, 512):
        raise ValueError(f"expected 512x512 mask, got {mask.shape}: {path}")
    return mask


def center_contained(crop_x: int, crop_y: int) -> bool:
    center_left = 256 - 10 // 2
    center_top = 256 - 10 // 2
    center_right = 256 + 10 // 2
    center_bottom = 256 + 10 // 2
    return (
        crop_x <= center_left
        and crop_y <= center_top
        and crop_x + PATCH_SIZE >= center_right
        and crop_y + PATCH_SIZE >= center_bottom
    )


def band_zero_too_much(array: np.ndarray) -> bool:
    band = array[BAND_INDEX]
    return float(np.count_nonzero(band == 0)) / band.size >= ZERO_RATIO_THRESHOLD


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".tmp.{os.getpid()}"
    )
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def write_tiff_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    tifffile.imwrite(temporary, array)
    os.replace(temporary, path)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stage_inputs(
    row: dict[str, Any],
    input_cache: Path,
) -> tuple[dict[str, Path], Path]:
    plume_id = str(row["plume_id"])
    plume_cache = input_cache / plume_id
    plume_cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for timepoint, (column, filename, _) in TIMEPOINTS.items():
        source = Path(str(row[column]))
        destination = plume_cache / filename
        if not destination.is_file() or destination.stat().st_size == 0:
            copy_atomic(source, destination)
        paths[timepoint] = destination
    mask_source = Path(str(row["resized_512x512_path"]))
    mask_destination = plume_cache / "plume_512.tif"
    if (
        not mask_destination.is_file()
        or mask_destination.stat().st_size == 0
    ):
        copy_atomic(mask_source, mask_destination)
    return paths, mask_destination


def sample_coordinates(
    plume_id: str,
    seed: int,
    positive_count: int,
    random_count: int,
) -> list[tuple[str, int, int, int]]:
    rng = random.Random(stable_seed(seed, plume_id))
    coordinates = [
        (
            "positive",
            index,
            rng.randint(234, 246),
            rng.randint(234, 246),
        )
        for index in range(positive_count)
    ]
    coordinates.extend(
        (
            "random",
            index,
            rng.randint(0, 512 - PATCH_SIZE),
            rng.randint(0, 512 - PATCH_SIZE),
        )
        for index in range(random_count)
    )
    return coordinates


def build_record(
    row: dict[str, Any],
    split: str,
    kind: str,
    index: int,
    crop_x: int,
    crop_y: int,
    label: int,
    mask_sum: float,
    local_sample_dir: Path,
    remote_sample_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample_id = (
        f"{row['plume_id']}__{kind}_{index:02d}_x{crop_x}_y{crop_y}"
    )
    common: dict[str, Any] = {
        "sample_id": sample_id,
        "id": sample_id,
        "plume_id": row["plume_id"],
        "event_group_id": row["event_group_id"],
        "split": split,
        "label": label,
        "crop_kind": kind,
        "crop_index": index,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "plume_mask_sum": mask_sum,
        "plume_latitude": row["plume_latitude"],
        "plume_longitude": row["plume_longitude"],
        "latitude": row["plume_latitude"],
        "longitude": row["plume_longitude"],
        "event_time": row["event_time"],
        "datetime": row["event_time"],
        "source": "legacy_notebook_cell7_six_timepoints",
        "image_center_mode": "plume_point",
    }

    def with_root(root: Path) -> dict[str, Any]:
        record = dict(common)
        for timepoint, (_, filename, time_column) in TIMEPOINTS.items():
            record[f"path_{timepoint}"] = str(root / filename)
            record[time_column] = row[time_column]
        mask_path = str(root / "plume.tif")
        record["path_plume"] = mask_path
        record["plume_mask_path"] = mask_path
        record["mask_path"] = mask_path
        record["mask_path_512"] = row["resized_512x512_path"]
        record["image_path"] = record["path_t0"]
        record["s2_path"] = record["path_t0"]
        record["s2_-7_path"] = record["path_prev1"]
        record["s2_pre_path"] = record["path_seasonal"]
        record["s2_pre_pre_path"] = record["path_year"]
        return record

    local_record = with_root(local_sample_dir)
    remote_record = (
        with_root(remote_sample_dir) if remote_sample_dir is not None else None
    )
    return local_record, remote_record


def crop_one(
    row: dict[str, Any],
    split: str,
    input_cache: Path,
    local_root: Path,
    remote_root: Path | None,
    positive_count: int,
    random_count: int,
    seed: int,
) -> dict[str, Any]:
    plume_id = str(row["plume_id"])
    local_plume_dir = local_root / split / plume_id
    local_manifest = local_plume_dir / "_manifest.json"
    if local_manifest.is_file() and local_manifest.stat().st_size > 0:
        data = json.loads(local_manifest.read_text())
        return {
            "plume_id": plume_id,
            "status": "local_exists",
            "local_records": data["local_records"],
            "remote_records": data.get("remote_records", []),
            "local_plume_dir": str(local_plume_dir),
        }

    staged_paths, staged_mask_path = stage_inputs(row, input_cache)
    arrays = {
        timepoint: to_chw(path)
        for timepoint, path in staged_paths.items()
    }
    for array in arrays.values():
        array[[8, 9]] = 0
    mask = load_mask(staged_mask_path)
    local_records: list[dict[str, Any]] = []
    remote_records: list[dict[str, Any]] = []
    rejected = 0
    for kind, index, crop_x, crop_y in sample_coordinates(
        plume_id,
        seed,
        positive_count,
        random_count,
    ):
        crops = {
            timepoint: array[
                :, crop_y : crop_y + PATCH_SIZE, crop_x : crop_x + PATCH_SIZE
            ]
            for timepoint, array in arrays.items()
        }
        if any(
            crop.shape != (12, PATCH_SIZE, PATCH_SIZE)
            or band_zero_too_much(crop)
            for crop in crops.values()
        ):
            rejected += 1
            continue
        is_positive = kind == "positive" or center_contained(crop_x, crop_y)
        label = int(is_positive)
        mask_crop = mask[
            crop_y : crop_y + PATCH_SIZE,
            crop_x : crop_x + PATCH_SIZE,
        ]
        if not is_positive:
            mask_crop = np.zeros(
                (PATCH_SIZE, PATCH_SIZE),
                dtype=mask.dtype,
            )

        sample_name = f"{kind}_{index:02d}_x{crop_x}_y{crop_y}"
        local_sample_dir = local_plume_dir / sample_name
        remote_sample_dir = (
            remote_root / split / plume_id / sample_name
            if remote_root is not None
            else None
        )
        for timepoint, (_, filename, _) in TIMEPOINTS.items():
            write_tiff_atomic(local_sample_dir / filename, crops[timepoint])
        write_tiff_atomic(local_sample_dir / "plume.tif", mask_crop)
        local_record, remote_record = build_record(
            row,
            split,
            kind,
            index,
            crop_x,
            crop_y,
            label,
            float(mask_crop.sum()),
            local_sample_dir,
            remote_sample_dir,
        )
        local_records.append(local_record)
        if remote_record is not None:
            remote_records.append(remote_record)

    if not local_records:
        raise ValueError("all sampled crops were rejected")
    manifest_data = {
        "plume_id": plume_id,
        "split": split,
        "rejected": rejected,
        "local_records": local_records,
        "remote_records": remote_records,
    }
    write_json_atomic(local_manifest, manifest_data)
    shutil.rmtree(input_cache / plume_id, ignore_errors=True)
    return {
        "plume_id": plume_id,
        "status": "cropped",
        "local_records": local_records,
        "remote_records": remote_records,
        "local_plume_dir": str(local_plume_dir),
        "rejected": rejected,
    }


def upload_one(
    local_plume_dir: str,
    remote_root: Path,
    split: str,
    plume_id: str,
) -> dict[str, Any]:
    source = Path(local_plume_dir)
    destination = remote_root / split / plume_id
    marker = destination / "_manifest.json"
    if marker.is_file() and marker.stat().st_size > 0:
        return {"plume_id": plume_id, "status": "remote_exists"}
    temporary = destination.with_name(
        destination.name + f".partial.{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    return {"plume_id": plume_id, "status": "uploaded"}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_split(
    split: str,
    csv_path: str,
    input_cache: Path,
    local_root: Path,
    remote_root: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame = pd.read_csv(csv_path, low_memory=False)
    frame = frame.sort_values(["event_time", "plume_id"], kind="stable")
    if args.limit > 0:
        frame = frame.head(args.limit)
    started = time.time()
    local_records: list[dict[str, Any]] = []
    remote_records: list[dict[str, Any]] = []
    crop_failures: list[dict[str, str]] = []
    upload_failures: list[dict[str, str]] = []
    upload_futures: dict[Future, str] = {}

    upload_pool = (
        ThreadPoolExecutor(max_workers=args.upload_workers)
        if remote_root is not None
        else None
    )
    with ThreadPoolExecutor(max_workers=args.workers) as crop_pool:
        crop_futures = {
            crop_pool.submit(
                crop_one,
                row,
                split,
                input_cache,
                local_root,
                remote_root,
                args.positive_count,
                args.random_count,
                args.seed,
            ): str(row["plume_id"])
            for row in frame.to_dict("records")
        }
        for index, future in enumerate(as_completed(crop_futures), start=1):
            plume_id = crop_futures[future]
            try:
                result = future.result()
                local_records.extend(result["local_records"])
                remote_records.extend(result["remote_records"])
                if upload_pool is not None:
                    upload_future = upload_pool.submit(
                        upload_one,
                        result["local_plume_dir"],
                        remote_root,
                        split,
                        plume_id,
                    )
                    upload_futures[upload_future] = plume_id
            except Exception as error:
                crop_failures.append(
                    {
                        "plume_id": plume_id,
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
            if index % args.progress_every == 0 or index == len(crop_futures):
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-6)
                eta_minutes = (
                    (len(crop_futures) - index) / max(rate, 1e-6) / 60.0
                )
                print(
                    f"[{split}] crop {index}/{len(crop_futures)} "
                    f"rows={len(local_records)} failures={len(crop_failures)} "
                    f"elapsed={elapsed / 60:.1f}m eta={eta_minutes:.1f}m",
                    flush=True,
                )
                atomic_csv(
                    pd.DataFrame(local_records),
                    local_root / f"{split}.csv",
                )

    if upload_pool is not None:
        for index, future in enumerate(as_completed(upload_futures), start=1):
            plume_id = upload_futures[future]
            try:
                future.result()
            except Exception as error:
                upload_failures.append(
                    {
                        "plume_id": plume_id,
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
            if index % args.progress_every == 0 or index == len(upload_futures):
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-6)
                eta_minutes = (
                    (len(upload_futures) - index)
                    / max(rate, 1e-6)
                    / 60.0
                )
                print(
                    f"[{split}] upload {index}/{len(upload_futures)} "
                    f"failures={len(upload_failures)} "
                    f"eta={eta_minutes:.1f}m",
                    flush=True,
                )
        upload_pool.shutdown()

    local_frame = pd.DataFrame(local_records).sort_values(
        ["event_group_id", "plume_id", "crop_kind", "crop_index"],
        kind="stable",
    )
    atomic_csv(local_frame, local_root / f"{split}.csv")
    if remote_root is not None:
        remote_frame = pd.DataFrame(remote_records).sort_values(
            ["event_group_id", "plume_id", "crop_kind", "crop_index"],
            kind="stable",
        )
        atomic_csv(remote_frame, remote_root / f"{split}.csv")
    return {
        "split": split,
        "input_plumes": len(frame),
        "cropped_plumes": len(frame) - len(crop_failures),
        "rows": len(local_frame),
        "labels": local_frame["label"].value_counts().sort_index().to_dict(),
        "crop_failures": crop_failures,
        "upload_failures": upload_failures,
        "seconds": round(time.time() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--remote-root", default="")
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--positive-count", type=int, default=16)
    parser.add_argument("--random-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--upload-workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    local_root = Path(args.local_root)
    remote_root = Path(args.remote_root) if args.remote_root else None
    input_cache = Path(args.input_cache)
    local_root.mkdir(parents=True, exist_ok=True)
    input_cache.mkdir(parents=True, exist_ok=True)
    if remote_root is not None:
        remote_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results = [
        run_split(
            "train",
            args.train_csv,
            input_cache,
            local_root,
            remote_root,
            args,
        ),
        run_split(
            "test",
            args.test_csv,
            input_cache,
            local_root,
            remote_root,
            args,
        ),
    ]
    train = pd.read_csv(local_root / "train.csv", low_memory=False)
    test = pd.read_csv(local_root / "test.csv", low_memory=False)
    overlap = sorted(
        set(train["event_group_id"].astype(str))
        & set(test["event_group_id"].astype(str))
    )
    if overlap:
        raise ValueError(f"event leakage detected: {overlap[:20]}")
    summary = {
        "seconds": round(time.time() - started, 2),
        "event_group_overlap": 0,
        "results": results,
        "args": vars(args),
    }
    write_json_atomic(local_root / "summary.json", summary)
    if remote_root is not None:
        write_json_atomic(remote_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    failed = sum(
        len(result["crop_failures"]) + len(result["upload_failures"])
        for result in results
    )
    if failed:
        raise RuntimeError(f"{failed} crop/upload failures")


if __name__ == "__main__":
    main()
