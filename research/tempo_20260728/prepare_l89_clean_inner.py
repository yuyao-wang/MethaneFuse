#!/usr/bin/env python3
"""Prepare and audit a train-only clean-L89 inner split.

This utility is intentionally limited to the explicitly supplied source
training CSV and existing ``/diniuvol`` local caches.  It never constructs,
lists, stats, or reads any test/sealed/holdout path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "Upgraded_dataset/l89_6time_full_event_balanced_split/"
    "L89_temporal_train_full_event_balanced.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "research/tempo_20260728/l89_clean_inner_v1"
)
LOCAL_3TIME = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_3time"
)
LOCAL_EXTRA = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_6time_extra"
)
STAGED_TRAIN = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/"
    "manifests_staged/l89_6time/train.csv"
)
STAGED_DEV = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/"
    "manifests_staged/l89_6time/val.csv"
)
HELD_OUT_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout|outer)([._-]|$)", re.IGNORECASE
)
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
ROLE_LAYOUT = {
    "path_t0": (LOCAL_3TIME, "l89_0.tif"),
    "path_prev1": (LOCAL_3TIME, "l89_prev1.tif"),
    "path_prev2": (LOCAL_EXTRA, "l89_prev2.tif"),
    "path_prev3": (LOCAL_EXTRA, "l89_prev3.tif"),
    "path_seasonal": (LOCAL_3TIME, "l89_seasonal.tif"),
    "path_year": (LOCAL_EXTRA, "l89_year.tif"),
}


def safe_train_path(value: str | Path, *, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    offending = [part for part in path.parts if HELD_OUT_RE.search(part)]
    if offending:
        raise ValueError(
            f"{purpose} contains held-out marker {sorted(offending)}: {path}"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_event_ids(frame: pd.DataFrame) -> pd.Series:
    if "plume_id" not in frame:
        raise ValueError("Source train CSV has no plume_id.")
    values = frame["plume_id"].astype("string").str.strip().str.replace(
        EVENT_SUFFIX_RE, "", regex=True
    )
    if values.isna().any() or values.eq("").any():
        raise ValueError("Canonical event derivation produced empty IDs.")
    return values.astype(str)


def choose_recent_development_events(
    frame: pd.DataFrame, *, fraction: float
) -> tuple[set[str], pd.Timestamp]:
    """Reuse the prior L89 policy: newest complete events to a row target."""

    summary = (
        frame.groupby("event_group_id", observed=True)
        .agg(
            event_time=("_event_time", "max"),
            rows=("event_group_id", "size"),
        )
        .sort_values(["event_time", "rows"], kind="mergesort")
    )
    target_rows = max(1, round(len(frame) * float(fraction)))
    chosen: list[str] = []
    selected_rows = 0
    for event_id, row in summary.iloc[::-1].iterrows():
        chosen.append(str(event_id))
        selected_rows += int(row["rows"])
        if selected_rows >= target_rows:
            break
    events = set(chosen)
    cutoff = summary.loc[list(events), "event_time"].min()
    return events, cutoff


def expected_local_paths(frame: pd.DataFrame) -> dict[str, list[Path]]:
    if "path" not in frame:
        raise ValueError("Source train CSV has no path column.")
    folders = frame["path"].astype(str).map(lambda value: Path(value).name)
    if folders.eq("").any():
        raise ValueError("At least one row has an empty source folder name.")
    return {
        role: [
            root / folder / filename
            for folder in folders.tolist()
        ]
        for role, (root, filename) in ROLE_LAYOUT.items()
    }


def parallel_is_file(
    paths: Sequence[Path], *, workers: int
) -> list[bool]:
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(Path.is_file, paths, chunksize=256))


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "events": int(frame["event_group_id"].nunique()),
        "labels": {
            str(int(label)): int(count)
            for label, count in frame["label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "time_min": frame["_event_time"].min().isoformat(),
        "time_max": frame["_event_time"].max().isoformat(),
    }


def run(args: argparse.Namespace) -> None:
    input_path = safe_train_path(args.input_csv, purpose="source train CSV")
    output_dir = safe_train_path(args.output_dir, purpose="inner split output")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frame = pd.read_csv(input_path, low_memory=False)
    required = {"id", "label", "plume_id", "path", "event_time", *ROLE_LAYOUT}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Source train CSV is missing {missing}.")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels are not binary.")
    frame["label"] = labels.astype(int)
    frame["event_group_id"] = canonical_event_ids(frame)
    frame["_event_time"] = pd.to_datetime(
        frame["event_time"], utc=True, errors="raise"
    )
    dev_events, cutoff = choose_recent_development_events(
        frame, fraction=float(args.dev_fraction)
    )
    dev = frame[frame["event_group_id"].isin(dev_events)].copy()
    train = frame[~frame["event_group_id"].isin(dev_events)].copy()
    overlap = set(train["event_group_id"]) & set(dev["event_group_id"])
    if overlap or train.empty or dev.empty:
        raise RuntimeError("Inner event split is invalid.")

    local_paths = expected_local_paths(frame)
    role_masks = {
        role: parallel_is_file(paths, workers=int(args.stat_workers))
        for role, paths in local_paths.items()
    }
    complete = pd.Series(True, index=frame.index)
    coverage_by_role: dict[str, Any] = {}
    for role, mask in role_masks.items():
        series = pd.Series(mask, index=frame.index)
        complete &= series
        coverage_by_role[role] = {
            "files": int(series.sum()),
            "rows": int(len(series)),
            "fraction": float(series.mean()),
        }
        frame[role] = [str(path) for path in local_paths[role]]
    frame["_local_six_role_complete"] = complete
    train = frame.loc[train.index].copy()
    dev = frame.loc[dev.index].copy()

    staged_ids: set[str] = set()
    staged_inputs: list[dict[str, Any]] = []
    for staged_path in (STAGED_TRAIN, STAGED_DEV):
        safe_path = safe_train_path(staged_path, purpose="safe staged manifest")
        staged = pd.read_csv(safe_path, usecols=["id"], low_memory=False)
        ids = set(staged["id"].astype(str))
        staged_ids.update(ids)
        staged_inputs.append(
            {
                "path": str(safe_path),
                "rows": int(len(staged)),
                "unique_ids": int(len(ids)),
                "sha256": sha256_file(safe_path),
            }
        )
    source_ids = frame["id"].astype(str)
    staged_covered = source_ids.isin(staged_ids)

    output_dir.mkdir(parents=True)
    helper_columns = ["_event_time"]
    output_columns = [
        column for column in frame.columns if column not in helper_columns
    ]
    train_output = output_dir / "train.csv"
    dev_output = output_dir / "dev.csv"
    train.loc[:, output_columns].to_csv(train_output, index=False)
    dev.loc[:, output_columns].to_csv(dev_output, index=False)

    complete_count = int(complete.sum())
    historical_seconds_per_row_one_gpu = (522.1 + 394.3) / (10033 + 9614)
    estimated_seconds = historical_seconds_per_row_one_gpu * len(frame)
    gate_fraction = float(args.minimum_local_coverage)
    local_gate = float(complete.mean()) >= gate_fraction
    audit = {
        "schema_version": "l89-clean-train-only-inner-readiness-v1",
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            **summarize(frame),
            "held_out_paths_read_or_statted_or_listed": False,
        },
        "event_rule": "plume_id:strip-final-hyphen-suffix",
        "inner_split": {
            "policy": "newest complete events until 20 percent row target",
            "development_target_fraction": float(args.dev_fraction),
            "development_cutoff_utc": cutoff.isoformat(),
            "train": summarize(train),
            "dev": summarize(dev),
            "event_overlap": int(len(overlap)),
            "train_csv": str(train_output),
            "dev_csv": str(dev_output),
        },
        "local_cache": {
            "roots_only": [str(LOCAL_3TIME), str(LOCAL_EXTRA)],
            "remote_source_paths_statted": False,
            "coverage_by_role": coverage_by_role,
            "six_role_complete_rows": complete_count,
            "six_role_missing_rows": int(len(frame) - complete_count),
            "six_role_complete_fraction": float(complete.mean()),
            "missing_id_sample": (
                frame.loc[~complete, "id"].astype(str).head(20).tolist()
            ),
            "staged_manifest_inputs": staged_inputs,
            "rows_already_present_in_prior_staged_manifests": int(
                staged_covered.sum()
            ),
            "prior_staged_manifest_row_fraction": float(
                staged_covered.mean()
            ),
        },
        "throughput_estimate": {
            "historical_train_rows": 10033,
            "historical_train_seconds": 522.1,
            "historical_dev_rows": 9614,
            "historical_dev_seconds": 394.3,
            "estimated_one_gpu_seconds_if_all_local": float(
                estimated_seconds
            ),
            "estimated_one_gpu_minutes_if_all_local": float(
                estimated_seconds / 60.0
            ),
            "conservative_1_5x_shared_gpu_minutes": float(
                estimated_seconds * 1.5 / 60.0
            ),
        },
        "readiness_gate": {
            "minimum_local_six_role_fraction": gate_fraction,
            "local_coverage_pass": bool(local_gate),
            "estimated_under_45_minutes_pass": bool(
                estimated_seconds * 1.5 < 45 * 60
            ),
            "gpu0_memory_gate_checked_externally": True,
            "ready_to_launch": False,
            "reason": (
                "GPU state must be appended after the audit."
                if local_gate
                else "local six-role coverage is below the required threshold"
            ),
        },
        "training_requirement": (
            "P5 and all D1 seeds must be retrained only on this inner train; "
            "prior heads overlap the new inner dev and are not reusable."
        ),
        "test_or_sealed_or_holdout_access": False,
    }
    audit_path = output_dir / "READINESS_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--stat-workers", type=int, default=32)
    parser.add_argument("--minimum-local-coverage", type=float, default=0.99)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
