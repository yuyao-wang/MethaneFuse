#!/usr/bin/env python3
"""Package HF dataset parts one at a time, upload each part, then delete it.

Example:
  HF_TOKEN=... python package_and_upload_hf_parts.py \
    --repo-id YOUR_NAME/YOUR_DATASET \
    --manifest hf_dataset_release/file_manifest.csv \
    --work-dir /diniuvol/yuyao/hf_dataset_parts_tmp \
    --max-uncompressed-gb 8
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, create_repo


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"uploaded_parts": [], "failed_parts": [], "missing_files": []}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_unique_manifest(path: Path, chunksize: int):
    seen_targets: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            source = str(row.source_path)
            target = str(row.target_path)
            if not source or not target or target in seen_targets:
                continue
            seen_targets.add(target)
            yield source, target


def upload_with_retry(api: HfApi, local_path: Path, repo_id: str, path_in_repo: str, repo_type: str, retries: int) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                sleep_s = min(60, 5 * attempt)
                print(f"upload failed for {local_path.name}, retry {attempt}/{retries}: {exc}; sleeping {sleep_s}s", flush=True)
                time.sleep(sleep_s)
    raise RuntimeError(f"upload failed after {retries} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. username/dataset_name")
    parser.add_argument("--manifest", default="hf_dataset_release/file_manifest.csv")
    parser.add_argument("--work-dir", default="/diniuvol/yuyao/hf_dataset_parts_tmp")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--path-prefix", default="", help="Optional prefix inside the HF repo, e.g. data_parts/")
    parser.add_argument("--max-uncompressed-gb", type=float, default=8.0)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--start-part", type=int, default=0, help="0 means resume from upload_state.json or start at 1")
    parser.add_argument("--skip-files", type=int, default=-1, help="Unique manifest rows to skip; -1 means resume from upload_state.json or 0")
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--delete-after-upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-parts", type=int, default=0, help="For testing; 0 means all parts")
    parser.add_argument("--dry-run", action="store_true", help="Build no tar and upload nothing; still stats source files")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN must be set for upload")

    manifest = Path(args.manifest)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "upload_state.json"
    missing_path = work_dir / "missing_files.csv"
    state = load_state(state_path)
    uploaded = set(state.get("uploaded_parts", []))
    skip_files = state.get("next_skip_files", 0) if args.skip_files < 0 else args.skip_files
    part_no = state.get("next_part", 1) if args.start_part == 0 else args.start_part

    api = HfApi(token=token) if token else None
    if args.create_repo and not args.dry_run:
        create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True, token=token)

    max_bytes = int(args.max_uncompressed_gb * 1024**3)
    current_size = 0
    current_count = 0
    total_size = 0
    total_count = 0
    part_file = None
    tar = None
    built_parts = 0

    def next_part_name(no: int) -> str:
        return f"dataset_part_{no:03d}.tar.gz"

    def open_part(no: int):
        name = next_part_name(no)
        path = work_dir / name
        if args.dry_run:
            return None, path
        return tarfile.open(path, "w:gz"), path

    def finish_part(no: int, path: Path, tar_obj, files: int, bytes_: int):
        nonlocal built_parts
        if files == 0:
            return
        if tar_obj is not None:
            tar_obj.close()
        name = path.name
        path_in_repo = f"{args.path_prefix.rstrip('/') + '/' if args.path_prefix else ''}{name}"
        print(json.dumps({"event": "part_ready", "part": no, "file": str(path), "files": files, "uncompressed_gb": bytes_ / 1024**3}), flush=True)
        if name in uploaded:
            print(f"skip already uploaded {name}", flush=True)
            if path.exists() and args.delete_after_upload:
                path.unlink()
        elif not args.dry_run:
            upload_with_retry(api, path, args.repo_id, path_in_repo, args.repo_type, args.retries)
            uploaded.add(name)
            state.setdefault("uploaded_parts", []).append(name)
            state["next_skip_files"] = skip_files + total_count
            state["next_part"] = no + 1
            save_state(state_path, state)
            print(f"uploaded {name} -> {args.repo_id}/{path_in_repo}", flush=True)
            if args.delete_after_upload and path.exists():
                path.unlink()
                print(f"deleted local {path}", flush=True)
        built_parts += 1

    if not args.dry_run:
        tar, part_file = open_part(part_no)

    missing_rows = []
    skipped = 0
    for source, target in iter_unique_manifest(manifest, args.chunksize):
        if skipped < skip_files:
            skipped += 1
            continue
        src = Path(source)
        try:
            size = src.stat().st_size
        except FileNotFoundError:
            missing_rows.append({"source_path": source, "target_path": target})
            continue

        if current_count and current_size + size > max_bytes:
            finish_part(part_no, part_file, tar, current_count, current_size)
            if args.max_parts and built_parts >= args.max_parts:
                break
            part_no += 1
            current_size = 0
            current_count = 0
            tar, part_file = open_part(part_no)

        if not args.dry_run:
            tar.add(src, arcname=target, recursive=False)
        current_size += size
        current_count += 1
        total_size += size
        total_count += 1

    else:
        finish_part(part_no, part_file, tar, current_count, current_size)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(missing_path, index=False)
        state["missing_files"] = str(missing_path)
        save_state(state_path, state)

    summary = {
        "repo_id": args.repo_id,
        "dry_run": args.dry_run,
        "skip_files_at_start": skip_files,
        "next_skip_files": skip_files + total_count,
        "next_part": part_no + 1,
        "total_seen_files": total_count,
        "total_seen_uncompressed_gb": total_size / 1024**3,
        "missing_files": len(missing_rows),
        "uploaded_parts": sorted(uploaded),
        "work_dir": str(work_dir),
    }
    (work_dir / "last_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
