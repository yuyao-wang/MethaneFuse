#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


PATH_COLUMNS = ("s2_0_path", "s2_90_path", "s2_360_path")
SPLITS = ("row_random_80_20", "event_disjoint_80_20")
PARTS = ("train", "test")


def sequence_hash(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for row in frame[list(columns)].itertuples(index=False, name=None):
        digest.update(("\0".join(map(str, row)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def file_status(path: str) -> tuple[str, bool, int]:
    candidate = Path(path)
    try:
        size = candidate.stat().st_size
    except OSError:
        return path, False, -1
    return path, size > 0, size


def verify_paths(paths: list[str], workers: int) -> dict[str, object]:
    unique_paths = sorted(set(paths))
    missing = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, valid, size in executor.map(file_status, unique_paths):
            if not valid:
                missing.append(path)
            elif size > 0:
                total_bytes += size
    return {
        "unique_paths": len(unique_paths),
        "valid_nonempty_paths": len(unique_paths) - len(missing),
        "missing_or_empty_paths": len(missing),
        "missing_or_empty_examples": missing[:20],
        "total_bytes": total_bytes,
    }


def build_part(
    new_csv: Path,
    old_by_id: pd.DataFrame,
    output_csv: Path,
    paired_new_csv: Path,
    invalid_legacy_ids: set[int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    new_frame = pd.read_csv(new_csv, low_memory=False)
    if new_frame["id"].duplicated().any():
        raise ValueError(f"duplicate ids in {new_csv}")
    dropped_ids = sorted(
        set(new_frame.loc[new_frame["id"].isin(invalid_legacy_ids), "id"].astype(int))
    )
    if dropped_ids:
        new_frame = new_frame.loc[~new_frame["id"].isin(dropped_ids)].reset_index(drop=True)
    old_rows = old_by_id.reindex(new_frame["id"].tolist())
    if old_rows[list(PATH_COLUMNS)].isna().any().any():
        missing_ids = new_frame.loc[
            old_rows[list(PATH_COLUMNS)].isna().any(axis=1).to_numpy(), "id"
        ].tolist()
        raise ValueError(f"legacy paths missing for {len(missing_ids)} ids: {missing_ids[:20]}")
    legacy_frame = new_frame.copy()
    for column in PATH_COLUMNS:
        legacy_frame[column] = old_rows[column].astype(str).to_numpy()
    if not legacy_frame[["id", "plume_id", "label", "event_id"]].equals(
        new_frame[["id", "plume_id", "label", "event_id"]]
    ):
        raise AssertionError("non-path sample identity changed")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paired_new_csv.parent.mkdir(parents=True, exist_ok=True)
    legacy_frame.to_csv(output_csv, index=False)
    new_frame.to_csv(paired_new_csv, index=False)
    audit = {
        "new_csv": str(new_csv),
        "paired_new_csv": str(paired_new_csv),
        "legacy_csv": str(output_csv),
        "rows": len(legacy_frame),
        "dropped_invalid_legacy_ids": dropped_ids,
        "plumes": int(legacy_frame["plume_id"].nunique()),
        "events": int(legacy_frame["event_id"].nunique()),
        "labels": {
            str(key): int(value)
            for key, value in legacy_frame["label"].value_counts().sort_index().items()
        },
        "sample_sequence_sha256": sequence_hash(
            legacy_frame, ("id", "plume_id", "label", "event_id")
        ),
        "paired_new_sample_sequence_sha256": sequence_hash(
            new_frame, ("id", "plume_id", "label", "event_id")
        ),
        "legacy_path_sequence_sha256": sequence_hash(legacy_frame, PATH_COLUMNS),
        "new_path_sequence_sha256": sequence_hash(new_frame, PATH_COLUMNS),
    }
    return legacy_frame, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-split-root", required=True)
    parser.add_argument("--legacy-cohort-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--paired-new-output-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args()

    old_frame = pd.read_csv(
        args.legacy_cohort_csv,
        usecols=["id", *PATH_COLUMNS],
        low_memory=False,
    )
    if old_frame["id"].duplicated().any():
        raise ValueError("duplicate ids in legacy cohort")
    old_by_id = old_frame.set_index("id")
    output_root = Path(args.output_root)
    paired_new_output_root = Path(args.paired_new_output_root)
    legacy_path_status = verify_paths(
        old_frame[list(PATH_COLUMNS)].to_numpy().reshape(-1).tolist(),
        args.workers,
    )
    invalid_paths = set(legacy_path_status["missing_or_empty_examples"])
    if legacy_path_status["missing_or_empty_paths"] > len(invalid_paths):
        raise RuntimeError(
            "invalid legacy path list was truncated; increase the audit capture before filtering"
        )
    invalid_legacy_ids = set(
        old_frame.loc[
            old_frame[list(PATH_COLUMNS)].isin(invalid_paths).any(axis=1), "id"
        ].astype(int)
    )
    report: dict[str, object] = {
        "new_split_root": args.new_split_root,
        "legacy_cohort_csv": args.legacy_cohort_csv,
        "output_root": args.output_root,
        "paired_new_output_root": args.paired_new_output_root,
        "legacy_cohort_file_verification": legacy_path_status,
        "invalid_legacy_ids_excluded_from_both_sides": sorted(invalid_legacy_ids),
        "splits": {},
    }
    all_paths: list[str] = []
    for split in SPLITS:
        split_report = {}
        for part in PARTS:
            legacy_frame, audit = build_part(
                Path(args.new_split_root) / split / f"{part}.csv",
                old_by_id,
                output_root / split / f"{part}.csv",
                paired_new_output_root / split / f"{part}.csv",
                invalid_legacy_ids,
            )
            all_paths.extend(
                legacy_frame[list(PATH_COLUMNS)].to_numpy().reshape(-1).tolist()
            )
            split_report[part] = audit
        report["splits"][split] = split_report
    if not args.skip_file_verification:
        report["file_verification"] = verify_paths(all_paths, args.workers)
        if report["file_verification"]["missing_or_empty_paths"]:
            raise RuntimeError(json.dumps(report["file_verification"], indent=2))
    audit_path = output_root / "manifest_audit.json"
    audit_path.write_text(json.dumps(report, indent=2) + "\n")
    (paired_new_output_root / "manifest_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
