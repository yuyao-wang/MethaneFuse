#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import s2_6time_cdse_legacy512_rebuild as pipeline


PATH_COLUMNS = [*pipeline.PATH_COLUMNS.values(), "path_plume"]
TRAINER_PATH_COLUMNS = list(pipeline.PATH_COLUMNS.values())
ALIAS_COLUMNS = [
    "plume_mask_path",
    "mask_path",
    "image_path",
    "s2_path",
    "s2_-7_path",
    "s2_pre_path",
    "s2_pre_pre_path",
]


def log(message: str) -> None:
    pipeline.log(f"upload32 {message}")


def atomic_state(done: dict[str, set[str]], path: Path) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "done": {split: sorted(values) for split, values in done.items()},
    }
    pipeline.atomic_json(payload, path)


def load_done(path: Path) -> dict[str, set[str]]:
    done = {"train": set(), "test": set()}
    if not path.exists() or path.stat().st_size <= 0:
        return done
    payload = json.loads(path.read_text(encoding="utf-8"))
    for split in done:
        done[split].update(map(str, payload.get("done", {}).get(split, [])))
    return done


def initialize_from_remote(done: dict[str, set[str]], remote_root: Path) -> None:
    for split in done:
        csv_path = remote_root / f"{split}_patches_32.csv"
        if not csv_path.exists() or csv_path.stat().st_size <= 0:
            continue
        frame = pd.read_csv(csv_path, usecols=["plume_id"], low_memory=False)
        done[split].update(frame["plume_id"].astype(str))
        log(f"initialized {split} done={len(done[split])} from {csv_path}")


def remote_path(source: str, local_root: Path, remote_root: Path) -> Path:
    path = Path(pipeline.clean(source))
    return remote_root / path.relative_to(local_root)


def unique_files(
    frame: pd.DataFrame,
    local_root: Path,
    remote_root: Path,
) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for column in PATH_COLUMNS:
        for value in frame[column].astype(str):
            source = Path(value)
            files[source] = remote_path(value, local_root, remote_root)
    return files


def prepare_remote_directories(
    destinations: list[Path], workers: int
) -> None:
    directories = sorted({destination.parent for destination in destinations})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda directory: directory.mkdir(parents=True, exist_ok=True), directories))


def copy_one(source: Path, destination: Path, buffer_size: int) -> tuple[bool, str]:
    try:
        if pipeline.file_ok(destination):
            return True, "exists"
        if not pipeline.file_ok(source):
            return False, f"missing_source:{source}"
        pipeline.copy_file_atomic(
            source,
            destination,
            buffer_size,
            create_parent=False,
        )
        if not pipeline.file_ok(destination):
            return False, f"invalid_destination:{destination}"
        return True, "copied"
    except Exception as error:
        return False, f"{type(error).__name__}:{error}"


def link_trainer_cache(
    source: Path,
    remote_source: Path,
    cache_root: Path,
) -> tuple[bool, str]:
    try:
        normalized = os.path.abspath(str(remote_source))
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        destination = cache_root / digest[:2] / f"{digest}{remote_source.suffix}"
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return True, "exists"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.part"
        )
        try:
            os.link(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return True, "linked"
    except Exception as error:
        return False, f"{type(error).__name__}:{error}"


def upload_batch(
    split: str,
    frame: pd.DataFrame,
    plume_ids: list[str],
    local_root: Path,
    remote_root: Path,
    workers: int,
    buffer_size: int,
    delete_local: bool,
    trainer_cache_root: Path | None,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    selected = frame[frame["plume_id"].astype(str).isin(plume_ids)].copy()
    plume_frames = {
        plume_id: group.copy()
        for plume_id, group in selected.groupby(selected["plume_id"].astype(str), sort=False)
    }
    plume_files = {
        plume_id: unique_files(group, local_root, remote_root)
        for plume_id, group in plume_frames.items()
    }
    all_destinations = [
        destination
        for files in plume_files.values()
        for destination in files.values()
    ]
    prepare_remote_directories(all_destinations, workers)

    failures: dict[str, list[str]] = {plume_id: [] for plume_id in plume_ids}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(copy_one, source, destination, buffer_size): (
                plume_id,
                source,
                destination,
            )
            for plume_id, files in plume_files.items()
            for source, destination in files.items()
        }
        for future in as_completed(futures):
            plume_id, source, destination = futures[future]
            valid, message = future.result()
            if not valid:
                failures[plume_id].append(
                    f"{message};source={source};destination={destination}"
                )

    completed = [plume_id for plume_id in plume_ids if not failures[plume_id]]
    failures = {plume_id: values for plume_id, values in failures.items() if values}
    cache_failures: dict[str, list[str]] = {}
    if trainer_cache_root is not None:
        trainer_files = {
            plume_id: {
                Path(source): remote_path(source, local_root, remote_root)
                for column in TRAINER_PATH_COLUMNS
                for source in plume_frames[plume_id][column].astype(str)
            }
            for plume_id in completed
        }
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    link_trainer_cache,
                    source,
                    destination,
                    trainer_cache_root,
                ): (plume_id, source, destination)
                for plume_id, files in trainer_files.items()
                for source, destination in files.items()
            }
            for future in as_completed(futures):
                plume_id, source, destination = futures[future]
                valid, message = future.result()
                if not valid:
                    cache_failures.setdefault(plume_id, []).append(
                        f"{message};source={source};remote_source={destination}"
                    )
    if delete_local:
        for plume_id in completed:
            shutil.rmtree(local_root / split / plume_id, ignore_errors=True)
    return completed, failures, cache_failures


def remap_frame(frame: pd.DataFrame, local_root: Path, remote_root: Path) -> pd.DataFrame:
    output = frame.copy()
    for column in [*PATH_COLUMNS, *ALIAS_COLUMNS]:
        if column not in output.columns:
            continue
        output[column] = [
            str(remote_path(value, local_root, remote_root))
            for value in output[column].astype(str)
        ]
    return output


def publish_metadata(local_root: Path, remote_root: Path) -> None:
    frames: list[pd.DataFrame] = []
    for split in ["train", "test"]:
        source_csv = local_root / f"{split}_patches_32.csv"
        frame = pd.read_csv(source_csv, low_memory=False)
        remapped = remap_frame(frame, local_root, remote_root)
        pipeline.atomic_csv(remapped, remote_root / f"{split}_patches_32.csv")
        frames.append(remapped)
        issue_csv = local_root / f"{split}_crop_issues.csv"
        if issue_csv.exists() and issue_csv.stat().st_size > 0:
            pipeline.copy_file_atomic(
                issue_csv,
                remote_root / issue_csv.name,
                8 * 1024 * 1024,
            )
    pipeline.atomic_csv(
        pd.concat(frames, ignore_index=True),
        remote_root / "all_patches_32.csv",
    )
    log(f"published final metadata to {remote_root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch-plumes", type=int, default=16)
    parser.add_argument("--buffer-mb", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--trainer-cache-dir", default="")
    parser.add_argument("--delete-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--initialize-from-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--limit-plumes", type=int, default=0)
    args = parser.parse_args()

    local_root = Path(args.local_root)
    remote_root = Path(args.remote_root)
    trainer_cache_root = (
        Path(args.trainer_cache_dir).expanduser().resolve()
        if pipeline.clean(args.trainer_cache_dir)
        else None
    )
    if trainer_cache_root is not None:
        trainer_cache_root.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_json)
    remote_root.mkdir(parents=True, exist_ok=True)
    done = load_done(state_path)
    if bool(args.initialize_from_remote) and not state_path.exists():
        initialize_from_remote(done, remote_root)
        atomic_state(done, state_path)

    while True:
        uploaded_this_scan = 0
        for split in ["train", "test"]:
            csv_path = local_root / f"{split}_patches_32.csv"
            if not csv_path.exists() or csv_path.stat().st_size <= 0:
                continue
            frame = pd.read_csv(csv_path, low_memory=False)
            available = list(dict.fromkeys(frame["plume_id"].astype(str)))
            already_done = done[split]
            for plume_id in available:
                if plume_id in already_done and bool(args.delete_local):
                    shutil.rmtree(local_root / split / plume_id, ignore_errors=True)
            pending = [plume_id for plume_id in available if plume_id not in already_done]
            if int(args.limit_plumes) > 0:
                pending = pending[: int(args.limit_plumes)]
            for start in range(0, len(pending), max(1, int(args.batch_plumes))):
                plume_ids = pending[start : start + max(1, int(args.batch_plumes))]
                completed, failures, cache_failures = upload_batch(
                    split,
                    frame,
                    plume_ids,
                    local_root,
                    remote_root,
                    int(args.workers),
                    max(1, int(args.buffer_mb)) * 1024 * 1024,
                    bool(args.delete_local),
                    trainer_cache_root,
                )
                done[split].update(completed)
                uploaded_this_scan += len(completed)
                atomic_state(done, state_path)
                log(
                    f"{split} uploaded={len(completed)}/{len(plume_ids)} "
                    f"done={len(done[split])} failures={len(failures)} "
                    f"trainer_cache_failures={len(cache_failures)}"
                )
                if failures:
                    failure_path = state_path.with_suffix(".failures.json")
                    pipeline.atomic_json(failures, failure_path)
                    raise RuntimeError(
                        f"upload failures={len(failures)}; details={failure_path}"
                    )
                if cache_failures:
                    failure_path = state_path.with_suffix(
                        ".trainer_cache_failures.json"
                    )
                    pipeline.atomic_json(cache_failures, failure_path)
                if bool(args.once) or int(args.limit_plumes) > 0:
                    break
            if bool(args.once) or int(args.limit_plumes) > 0:
                break

        all_csv = local_root / "all_patches_32.csv"
        if all_csv.exists() and all_csv.stat().st_size > 0:
            complete = True
            for split in ["train", "test"]:
                csv_path = local_root / f"{split}_patches_32.csv"
                frame = pd.read_csv(csv_path, usecols=["plume_id"], low_memory=False)
                expected = set(frame["plume_id"].astype(str))
                complete = complete and expected.issubset(done[split])
            if complete:
                publish_metadata(local_root, remote_root)
                log("all local crop32 files uploaded and removed")
                return 0
        if bool(args.once) or int(args.limit_plumes) > 0:
            return 0
        if uploaded_this_scan == 0:
            time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
