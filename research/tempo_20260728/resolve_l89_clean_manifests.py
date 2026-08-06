#!/usr/bin/env python3
"""Resolve the train-only clean L89 split onto audited local storage.

The input is restricted to the two explicit ``train``/``dev`` CSVs produced
by :mod:`prepare_l89_clean_inner`.  Existing path columns are deliberately
ignored: every image path is reconstructed from the source folder basename
and one of three fixed, local allow-listed roots.

For each role, the historical local cache has priority.  A missing historical
file falls back to the new full-local staging root.  Formal manifests are
written only if every declared role file exists and train/development
canonical events are disjoint.  This program never constructs, lists, stats,
or reads a test/sealed/holdout/outer path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "research/tempo_20260728/l89_clean_inner_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260728/"
    "l89_clean_replicate_v1/manifests"
)
DEFAULT_OLD_3TIME = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_3time"
)
DEFAULT_OLD_EXTRA = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/cache/l89_6time_extra"
)
DEFAULT_STAGING_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260728/l89_clean_full_local_v1"
)

FORBIDDEN_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout|outer)([._/\\-]|$)", re.IGNORECASE
)
FOLDER_RE = re.compile(r"^[0-9]{8}$")
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
ROLE_FILES: Mapping[str, tuple[str, str]] = {
    "path_t0": ("old_3time", "l89_0.tif"),
    "path_prev1": ("old_3time", "l89_prev1.tif"),
    "path_prev2": ("old_extra", "l89_prev2.tif"),
    "path_prev3": ("old_extra", "l89_prev3.tif"),
    "path_seasonal": ("old_3time", "l89_seasonal.tif"),
    "path_year": ("old_extra", "l89_year.tif"),
}
EXPECTED_ROWS = {"train": 18_682, "dev": 4_677}
EXPECTED_EVENTS = {"train": 672, "dev": 176}
EXPECTED_FROZEN_INPUT_SHA256: Mapping[str, str] = {
    "train": "8cee38e78f0ff7ebf5e34c02ba07d351cbf93438d9a78e88ad741dc8be3a22eb",
    "dev": "8caa5f7a31340c5007ec4457fd5766179a23dc1e97b432322768a8cef0144883",
}
READINESS_FILENAME = "READINESS_AUDIT.json"
EXPECTED_READINESS_SHA256 = (
    "da54079d348d3d9cf13d3609a61e6315169e75d8b8ee977caa8143901c7c69df"
)


class IncompleteLocalStaging(RuntimeError):
    """Raised before any formal manifest is written when local files are absent."""


def safe_path(value: str | Path, *, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    if FORBIDDEN_RE.search(str(path)):
        raise ValueError(f"{purpose} contains a held-out marker: {path}")
    return path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_split_inputs(input_root: Path) -> dict[str, Any]:
    """Reject any split payload other than the protocol-locked inner split."""

    paths = {
        "train": input_root / "train.csv",
        "dev": input_root / "dev.csv",
        "readiness": input_root / READINESS_FILENAME,
    }
    expected = {
        **EXPECTED_FROZEN_INPUT_SHA256,
        "readiness": EXPECTED_READINESS_SHA256,
    }
    receipt: dict[str, Any] = {}
    for name in ("train", "dev", "readiness"):
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha256 = sha256_file(path)
        expected_sha256 = expected[name]
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"Frozen clean-inner {name} SHA-256 mismatch: "
                f"observed={observed_sha256}, expected={expected_sha256}, "
                f"path={path}"
            )
        receipt[name] = {
            "path": str(path),
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "exact_sha_match": True,
        }
    return {
        "schema_version": "l89-clean-inner-split-lock-v1",
        "enforced": True,
        "different_split_rejected": True,
        "files": receipt,
    }


def atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def canonical_event_ids(frame: pd.DataFrame) -> pd.Series:
    values = (
        frame["plume_id"]
        .astype("string")
        .str.strip()
        .str.replace(EVENT_SUFFIX_RE, "", regex=True)
    )
    if values.isna().any() or values.eq("").any():
        raise ValueError("Canonical event derivation produced an empty value.")
    return values.astype(str)


def folder_names(frame: pd.DataFrame) -> list[str]:
    folders = [Path(value).name for value in frame["path"].astype(str)]
    invalid = sorted({value for value in folders if not FOLDER_RE.fullmatch(value)})
    if invalid:
        raise ValueError(
            "Source folder basenames must be exactly eight digits; "
            f"examples={invalid[:10]}"
        )
    return folders


def resolve_one(
    item: tuple[str, str, Path, Path]
) -> tuple[Optional[str], str]:
    folder, filename, historical_root, staging_root = item
    historical = historical_root / folder / filename
    if historical.is_file():
        return str(historical), "historical"
    staged = staging_root / folder / filename
    if staged.is_file():
        return str(staged), "staged"
    return None, "missing"


def validate_input_frame(
    frame: pd.DataFrame, *, split: str, enforce_expected_counts: bool
) -> pd.DataFrame:
    required = {
        "id",
        "label",
        "plume_id",
        "path",
        "event_group_id",
        *ROLE_FILES,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{split} CSV is missing columns {missing}.")
    if frame.empty:
        raise ValueError(f"{split} CSV is empty.")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"{split} CSV contains duplicate IDs.")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{split} labels are not binary.")
    frame = frame.copy()
    frame["label"] = labels.astype(int)
    derived = canonical_event_ids(frame)
    declared = frame["event_group_id"].astype(str)
    if not declared.equals(derived):
        raise ValueError(
            f"{split} event_group_id differs from canonical plume_id derivation."
        )
    if enforce_expected_counts:
        if len(frame) != EXPECTED_ROWS[split]:
            raise ValueError(
                f"{split} rows {len(frame)} != frozen {EXPECTED_ROWS[split]}."
            )
        events = int(declared.nunique())
        if events != EXPECTED_EVENTS[split]:
            raise ValueError(
                f"{split} events {events} != frozen {EXPECTED_EVENTS[split]}."
            )
    return frame


def resolve_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    old_3time: Path,
    old_extra: Path,
    staging_root: Path,
    workers: int,
    enforce_expected_counts: bool,
    smoke_rows: int = 0,
    smoke_seed: int = 20260728,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    roots = {
        "old_3time": safe_path(old_3time, purpose="historical three-time root"),
        "old_extra": safe_path(old_extra, purpose="historical extra-time root"),
    }
    staging_root = safe_path(staging_root, purpose="new staging root")
    validated = {
        split: validate_input_frame(
            frame, split=split, enforce_expected_counts=enforce_expected_counts
        )
        for split, frame in frames.items()
    }
    overlap = set(validated["train"]["event_group_id"].astype(str)) & set(
        validated["dev"]["event_group_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"Train/development overlap by {len(overlap)} canonical events; "
            f"examples={sorted(overlap)[:10]}"
        )

    resolved: dict[str, pd.DataFrame] = {}
    source_counts: dict[str, Any] = {}
    missing_samples: list[dict[str, str]] = []
    missing_total = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        for split in ("train", "dev"):
            frame = validated[split].copy()
            folders = folder_names(frame)
            source_counts[split] = {}
            for column, (root_name, filename) in ROLE_FILES.items():
                requests = [
                    (folder, filename, roots[root_name], staging_root)
                    for folder in folders
                ]
                answers = list(
                    executor.map(resolve_one, requests, chunksize=256)
                )
                paths = [answer[0] for answer in answers]
                origins = [answer[1] for answer in answers]
                counts = {
                    name: int(sum(origin == name for origin in origins))
                    for name in ("historical", "staged", "missing")
                }
                source_counts[split][column] = counts
                for row_index, (path, origin) in enumerate(zip(paths, origins)):
                    if origin == "missing":
                        missing_total += 1
                        if len(missing_samples) < 30:
                            missing_samples.append(
                                {
                                    "split": split,
                                    "id": str(frame.iloc[row_index]["id"]),
                                    "folder": folders[row_index],
                                    "role": column,
                                }
                            )
                frame[column] = [value or "" for value in paths]
            resolved[split] = frame

    complete_rows_by_split: dict[str, int] = {}
    if smoke_rows:
        if smoke_rows < 1:
            raise ValueError("smoke_rows must be positive.")
        for split in ("train", "dev"):
            frame = resolved[split]
            complete = frame[list(ROLE_FILES)].ne("").all(axis=1)
            complete_rows_by_split[split] = int(complete.sum())
            candidates = frame.loc[complete].copy()
            if len(candidates) < smoke_rows:
                raise IncompleteLocalStaging(
                    f"{split} has only {len(candidates)} fully local rows; "
                    f"{smoke_rows} are required for the memory smoke."
                )
            # Stable pseudo-random order avoids making the smoke depend on
            # whichever events happen to appear first in the source CSV.
            keys = candidates["id"].astype(str).map(
                lambda value: hashlib.sha256(
                    f"{int(smoke_seed)}:{value}".encode("utf-8")
                ).hexdigest()
            )
            candidates = (
                candidates.assign(_smoke_order=keys)
                .sort_values(["_smoke_order", "id"], kind="mergesort")
                .head(int(smoke_rows))
                .drop(columns=["_smoke_order"])
                .reset_index(drop=True)
            )
            if not candidates[list(ROLE_FILES)].applymap(
                lambda value: Path(str(value)).is_file()
            ).all(axis=None):
                raise RuntimeError(
                    f"{split} selected smoke rows changed during final existence check."
                )
            resolved[split] = candidates

    audit = {
        "schema_version": (
            "l89-clean-resolved-smoke-manifests-v1"
            if smoke_rows
            else "l89-clean-resolved-manifests-v1"
        ),
        "inputs": {
            split: {
                "rows": int(len(validated[split])),
                "events": int(
                    validated[split]["event_group_id"].astype(str).nunique()
                ),
            }
            for split in ("train", "dev")
        },
        "event_overlap": 0,
        "role_source_counts": source_counts,
        "missing_role_files": int(missing_total),
        "missing_sample": missing_samples,
        "roots": {
            "historical_three_time": str(roots["old_3time"]),
            "historical_extra_time": str(roots["old_extra"]),
            "new_staging": str(staging_root),
        },
        "path_resolution_policy": (
            "historical role-specific local cache first; fixed new staging "
            "root second; existing CSV path_* values ignored"
        ),
        "remote_source_path_read_or_statted": False,
        "held_out_path_constructed_or_listed_or_statted_or_read": False,
        "complete": bool(missing_total == 0),
        "formal_full_manifest": bool(smoke_rows == 0),
        "smoke": (
            {
                "rows_per_split": int(smoke_rows),
                "selection_seed": int(smoke_seed),
                "fully_local_candidates": complete_rows_by_split,
                "every_selected_role_file_exists": True,
            }
            if smoke_rows
            else None
        ),
    }
    if missing_total and not smoke_rows:
        raise IncompleteLocalStaging(json.dumps(audit, sort_keys=True))
    return resolved, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_root = safe_path(args.input_root, purpose="clean inner input root")
    output_root = safe_path(args.output_root, purpose="resolved manifest output root")
    frozen_split_lock = validate_frozen_split_inputs(input_root)
    inputs = {
        "train": safe_path(input_root / "train.csv", purpose="train CSV"),
        "dev": safe_path(input_root / "dev.csv", purpose="development CSV"),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frames = {
        split: pd.read_csv(path, low_memory=False)
        for split, path in inputs.items()
    }
    resolved, audit = resolve_frames(
        frames,
        old_3time=Path(args.old_3time_root),
        old_extra=Path(args.old_extra_root),
        staging_root=Path(args.staging_root),
        workers=args.workers,
        enforce_expected_counts=not args.no_frozen_count_check,
        smoke_rows=args.smoke_rows,
        smoke_seed=args.smoke_seed,
    )
    audit["inputs"] = {
        split: {
            **audit["inputs"][split],
            "path": str(inputs[split]),
            "sha256": sha256_file(inputs[split]),
        }
        for split in ("train", "dev")
    }
    audit["frozen_split_lock"] = frozen_split_lock
    audit["output_root"] = str(output_root)
    if args.check_only:
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
        return audit

    outputs = {
        "train": output_root / "train.csv",
        "dev": output_root / "dev.csv",
    }
    collisions = [path for path in (*outputs.values(), output_root / "AUDIT.json") if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {collisions}.")
    for split in ("train", "dev"):
        atomic_csv_write(outputs[split], resolved[split])
    audit["outputs"] = {
        split: {
            "path": str(outputs[split]),
            "sha256": sha256_file(outputs[split]),
            "rows": int(len(resolved[split])),
            "events": int(resolved[split]["event_group_id"].nunique()),
        }
        for split in ("train", "dev")
    }
    atomic_json_write(output_root / "AUDIT.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--old-3time-root", default=str(DEFAULT_OLD_3TIME))
    parser.add_argument("--old-extra-root", default=str(DEFAULT_OLD_EXTRA))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help=(
            "Write a deterministic fully-local subset per split even while "
            "unselected staging rows remain incomplete. Zero means require "
            "the complete formal cohort."
        ),
    )
    parser.add_argument("--smoke-seed", type=int, default=20260728)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--no-frozen-count-check",
        action="store_true",
        help="Testing only: do not require the frozen production row/event counts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
