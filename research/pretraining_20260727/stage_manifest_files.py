#!/usr/bin/env python3
"""Stage path-backed manifest rows onto fast local storage.

The source manifests may contain paths under an obsolete cache prefix.  This
tool maps those paths to a surviving source tree, copies the selected files in
parallel, and writes manifests whose path columns point at the new cache.
Copies are atomic and existing files are accepted only when their sizes match.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


DEFAULT_PATH_COLUMNS = (
    "path_t0",
    "path_prev1",
    "path_prev2",
    "path_prev3",
    "path_seasonal",
    "path_year",
)


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL=/absolute/path.csv, received {value!r}"
        )
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Manifest label cannot be empty")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"Manifest path must be absolute: {path}")
    return label, path


def parse_columns(value: str) -> tuple[str, ...]:
    columns = tuple(part.strip() for part in value.split(",") if part.strip())
    if not columns:
        raise argparse.ArgumentTypeError("At least one path column is required")
    return columns


def read_rows(path: Path, limit_rows: int | None) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = []
        for row_index, row in enumerate(reader):
            if limit_rows is not None and row_index >= limit_rows:
                break
            rows.append(dict(row))
    return list(reader.fieldnames), rows


def mapped_paths(
    manifest_path: str,
    manifest_prefix: Path,
    source_prefix: Path,
    destination_prefix: Path,
) -> tuple[Path, Path]:
    raw_path = Path(manifest_path)
    try:
        relative = raw_path.relative_to(manifest_prefix)
    except ValueError as exc:
        raise ValueError(
            f"Path {raw_path} is outside manifest prefix {manifest_prefix}"
        ) from exc
    return source_prefix / relative, destination_prefix / relative


def copy_one(source: Path, destination: Path) -> tuple[str, int]:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_size = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == source_size:
        return "cached", source_size

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.part.{os.getpid()}.{os.getpid() ^ hash(destination)}"
    )
    try:
        shutil.copyfile(source, temporary)
        copied_size = temporary.stat().st_size
        if copied_size != source_size:
            raise OSError(
                f"Size mismatch after copy: {source} ({source_size}) -> "
                f"{temporary} ({copied_size})"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "copied", source_size


def write_manifest(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        type=parse_manifest_arg,
        metavar="LABEL=/ABS/PATH.csv",
    )
    parser.add_argument("--manifest_prefix", type=Path, required=True)
    parser.add_argument("--source_prefix", type=Path, required=True)
    parser.add_argument("--destination_prefix", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--path_columns",
        type=parse_columns,
        default=DEFAULT_PATH_COLUMNS,
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit_rows", type=int)
    parser.add_argument("--progress_every", type=int, default=10_000)
    args = parser.parse_args()

    manifest_prefix = args.manifest_prefix.resolve()
    source_prefix = args.source_prefix.resolve()
    destination_prefix = args.destination_prefix.resolve()
    path_columns = tuple(args.path_columns)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.limit_rows is not None and args.limit_rows < 1:
        raise ValueError("--limit_rows must be positive")

    outputs: list[tuple[str, list[str], list[dict[str, str]], Path]] = []
    jobs: dict[Path, Path] = {}
    for label, manifest_path in args.manifest:
        fieldnames, rows = read_rows(manifest_path, args.limit_rows)
        missing_columns = [column for column in path_columns if column not in fieldnames]
        if missing_columns:
            raise ValueError(f"{manifest_path} is missing path columns {missing_columns}")
        for row in rows:
            for column in path_columns:
                source, destination = mapped_paths(
                    row[column],
                    manifest_prefix,
                    source_prefix,
                    destination_prefix,
                )
                previous = jobs.setdefault(destination, source)
                if previous != source:
                    raise ValueError(
                        f"Destination collision: {destination} maps to both "
                        f"{previous} and {source}"
                    )
                row[column] = str(destination)
        output_path = args.output_dir / f"{label}.csv"
        outputs.append((label, fieldnames, rows, output_path))

    copied_files = 0
    cached_files = 0
    staged_bytes = 0
    completed = 0
    print(
        f"[stage] manifests={len(outputs)} unique_files={len(jobs)} "
        f"workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(copy_one, source, destination): (source, destination)
            for destination, source in jobs.items()
        }
        for future in as_completed(futures):
            source, destination = futures[future]
            try:
                state, size = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed staging {source} -> {destination}"
                ) from exc
            completed += 1
            staged_bytes += size
            copied_files += int(state == "copied")
            cached_files += int(state == "cached")
            if completed % args.progress_every == 0 or completed == len(jobs):
                print(
                    f"[stage] {completed}/{len(jobs)} files; "
                    f"copied={copied_files}, cached={cached_files}, "
                    f"logical_GiB={staged_bytes / 2**30:.2f}",
                    flush=True,
                )

    manifest_rows: dict[str, int] = {}
    for label, fieldnames, rows, output_path in outputs:
        write_manifest(output_path, fieldnames, rows)
        manifest_rows[label] = len(rows)
        print(f"[stage] wrote {output_path} ({len(rows)} rows)", flush=True)

    audit = {
        "manifest_rows": manifest_rows,
        "unique_files": len(jobs),
        "copied_files": copied_files,
        "cached_files": cached_files,
        "logical_bytes": staged_bytes,
        "manifest_prefix": str(manifest_prefix),
        "source_prefix": str(source_prefix),
        "destination_prefix": str(destination_prefix),
        "path_columns": list(path_columns),
        "limit_rows": args.limit_rows,
    }
    audit_path = args.output_dir / "stage_audit.json"
    temporary_audit = audit_path.with_suffix(
        audit_path.suffix + f".part.{os.getpid()}"
    )
    temporary_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_audit, audit_path)


if __name__ == "__main__":
    main()
