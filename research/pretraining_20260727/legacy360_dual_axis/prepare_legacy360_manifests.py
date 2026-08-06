#!/usr/bin/env python3
"""Prepare full legacy-360 m manifests for time x sensor experiments.

The split selection is deliberately train-only:

* enrich the user-specified legacy train/test manifests with the existing
  geographic cluster metadata by globally unique ``id``;
* derive a canonical acquisition event by stripping the terminal plume
  variant (for example ``...-A``);
* select approximately ten percent of *train* as development data while
  keeping complete canonical events together and matching label/sensor-combo
  distributions;
* retain every source row (there is no representative-row subsampling);
* compute S5P availability solely from ``s5p_0_path``.  That path is an NPZ
  containing the temporal triplet; the legacy ``s5p_90_path`` and
  ``s5p_360_path`` columns are intentionally ignored for availability.

No raster or NPZ payload is opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_LEGACY_ROOT = Path(
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "finalDataset_query/legacy_param_360m"
)
DEFAULT_METADATA_ROOT = Path(
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "Dataset/datasets_360m_cluster_split"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "manifests"
)

SENSORS = ("s2", "l89", "emit", "s5p")
OPTICAL_SENSORS = ("s2", "l89", "emit")
OPTICAL_ROLES = ("0", "90", "360")
OPTICAL_COLUMNS = {
    sensor: tuple(f"{sensor}_{role}_path" for role in OPTICAL_ROLES)
    for sensor in OPTICAL_SENSORS
}
CLASSIFICATION_PATH_COLUMNS = (
    *(column for sensor in OPTICAL_SENSORS for column in OPTICAL_COLUMNS[sensor]),
    "s5p_0_path",
)
REQUIRED_SOURCE_COLUMNS = (
    "id",
    "plume_id",
    "label",
    *CLASSIFICATION_PATH_COLUMNS,
)
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
EMPTY_TOKENS = {"", "nan", "none", "null"}
UINT64_MASK = (1 << 64) - 1


def clean_series(series: pd.Series) -> pd.Series:
    result = series.astype(str).str.strip()
    return result.mask(result.str.casefold().isin(EMPTY_TOKENS), "")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], role: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{role} is missing columns: {missing}")


def read_source(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path, dtype=str, keep_default_na=False, low_memory=False
    )
    ensure_columns(frame, REQUIRED_SOURCE_COLUMNS, role)
    if frame.empty:
        raise ValueError(f"{role} is empty: {path}")
    for column in ("id", "plume_id"):
        frame[column] = clean_series(frame[column])
        if frame[column].eq("").any():
            raise ValueError(f"{role}.{column} contains empty values")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{role}.label must contain only integer 0/1")
    frame["label"] = labels.astype(np.int64)
    return frame


def read_metadata(paths: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    required = ("id", "_sample_id", "cluster_id", "macro_region_id")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        piece = pd.read_csv(
            path,
            usecols=list(required),
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        ensure_columns(piece, required, f"metadata {path}")
        pieces.append(piece)
    metadata = pd.concat(pieces, ignore_index=True)
    for column in required:
        metadata[column] = clean_series(metadata[column])
        if metadata[column].eq("").any():
            raise ValueError(f"metadata column {column!r} contains blanks")
    if metadata["id"].duplicated().any():
        duplicates = metadata.loc[
            metadata["id"].duplicated(keep=False), "id"
        ].head(20)
        raise ValueError(f"metadata ids are not unique: {duplicates.tolist()}")
    return metadata


def canonical_event_ids(plume_ids: pd.Series) -> pd.Series:
    events = plume_ids.astype(str).str.replace(
        EVENT_SUFFIX_RE, "", regex=True
    )
    if events.eq("").any():
        raise ValueError("canonical event derivation produced an empty value")
    return events


def add_availability(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    result = frame.copy()
    available: dict[str, np.ndarray] = {}
    for sensor in OPTICAL_SENSORS:
        present = np.stack(
            [
                clean_series(result[column]).ne("").to_numpy(dtype=bool)
                for column in OPTICAL_COLUMNS[sensor]
            ],
            axis=1,
        )
        partial = present.any(axis=1) & ~present.all(axis=1)
        if partial.any():
            examples = np.flatnonzero(partial)[:20].tolist()
            raise ValueError(
                f"{role} has partial {sensor} triplets at rows {examples}"
            )
        available[sensor] = present.all(axis=1)

    # One s5p_0 NPZ stores all temporal roles.  Never infer availability from
    # the legacy s5p_90/s5p_360 columns.
    s5p_paths = clean_series(result["s5p_0_path"])
    invalid_s5p = s5p_paths.ne("") & ~s5p_paths.str.casefold().str.endswith(".npz")
    if invalid_s5p.any():
        examples = s5p_paths[invalid_s5p].head(20).tolist()
        raise ValueError(
            f"{role} has non-NPZ s5p_0 paths (examples={examples})"
        )
    available["s5p"] = s5p_paths.ne("").to_numpy(dtype=bool)

    matrix = np.stack([available[sensor] for sensor in SENSORS], axis=1)
    signatures = [
        "+".join(
            sensor for sensor, is_available in zip(SENSORS, row) if is_available
        )
        for row in matrix.tolist()
    ]
    if any(not value for value in signatures):
        rows = [index for index, value in enumerate(signatures) if not value][:20]
        raise ValueError(f"{role} has rows with no available sensor: {rows}")
    result["availability_signature"] = signatures
    return result


def enrich_sources(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_ids = pd.concat(
        [train["id"], evaluation["id"]], ignore_index=True
    )
    if source_ids.duplicated().any():
        duplicates = source_ids[source_ids.duplicated(keep=False)].head(20)
        raise ValueError(
            "legacy train/evaluation ids are not globally disjoint: "
            f"{duplicates.tolist()}"
        )
    if set(source_ids) != set(metadata["id"]):
        missing_metadata = sorted(set(source_ids) - set(metadata["id"]))[:20]
        extra_metadata = sorted(set(metadata["id"]) - set(source_ids))[:20]
        raise ValueError(
            "source/metadata id universes differ: "
            f"missing_metadata={missing_metadata}, "
            f"extra_metadata={extra_metadata}"
        )

    mapping = metadata.set_index("id")

    def enrich(frame: pd.DataFrame, role: str, offset: int) -> pd.DataFrame:
        joined = frame.merge(
            mapping.reset_index(),
            on="id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if not joined["_sample_id"].eq(joined["plume_id"]).all():
            bad = joined.loc[
                ~joined["_sample_id"].eq(joined["plume_id"]),
                ["id", "plume_id", "_sample_id"],
            ].head(20)
            raise ValueError(
                f"{role} metadata id-to-plume mismatch: "
                f"{bad.to_dict(orient='records')}"
            )
        joined["event_id"] = canonical_event_ids(joined["plume_id"])
        joined["query360_index"] = np.arange(
            offset, offset + len(joined), dtype=np.int64
        )
        joined = add_availability(joined, role)
        return joined

    return (
        enrich(train, "train", 0),
        enrich(evaluation, "evaluation", len(train)),
    )


def total_variation(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(0.5 * np.abs(candidate - reference).sum())


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized, explicitly wrapping SplitMix64 permutation."""

    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return values ^ (values >> np.uint64(31))


def choose_dev_events(
    train: pd.DataFrame,
    *,
    seed: int,
    fraction: float,
    search_trials: int,
) -> tuple[set[str], dict[str, Any]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError(f"dev fraction must be in (0, 0.5), got {fraction}")
    if search_trials < 1:
        raise ValueError("search_trials must be positive")

    work = train[["event_id", "label", "availability_signature"]].copy()
    work["_label"] = work["label"].astype(str)
    work["_stratum"] = (
        work["_label"] + "|" + work["availability_signature"].astype(str)
    )
    event_ids = sorted(work["event_id"].unique().tolist())
    if len(event_ids) < 10:
        raise ValueError("too few canonical events for a ten-percent dev split")

    row_counts = (
        work.groupby("event_id", observed=True)
        .size()
        .reindex(event_ids, fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    strata = sorted(work["_stratum"].unique().tolist())
    signatures = sorted(work["availability_signature"].unique().tolist())
    labels = sorted(work["_label"].unique().tolist())

    stratum_counts = (
        pd.crosstab(work["event_id"], work["_stratum"])
        .reindex(index=event_ids, columns=strata, fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    signature_counts = (
        pd.crosstab(work["event_id"], work["availability_signature"])
        .reindex(index=event_ids, columns=signatures, fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    label_counts = (
        pd.crosstab(work["event_id"], work["_label"])
        .reindex(index=event_ids, columns=labels, fill_value=0)
        .to_numpy(dtype=np.int64)
    )

    total_rows = int(row_counts.sum())
    target_rows = float(fraction * total_rows)
    global_stratum = stratum_counts.sum(axis=0) / total_rows
    global_signature = signature_counts.sum(axis=0) / total_rows
    global_label = label_counts.sum(axis=0) / total_rows
    # A combo represented by at least five independent events must occur on
    # both sides.  The analogous threshold is ten events for the finer
    # label-by-combo strata.  Rarer groups remain in the distribution
    # objective but cannot safely be made a hard constraint.
    constrained_signatures = (signature_counts > 0).sum(axis=0) >= 5
    constrained_strata = (stratum_counts > 0).sum(axis=0) >= 10
    base_hashes = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{seed}\0{event_id}".encode("utf-8")
                ).digest()[:8],
                byteorder="little",
                signed=False,
            )
            for event_id in event_ids
        ],
        dtype=np.uint64,
    )

    best_key: tuple[float, str] | None = None
    best_indices: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    for trial in range(search_trials):
        salt = np.uint64(
            ((trial + 1) * 0xD1B54A32D192ED03) & UINT64_MASK
        )
        keys = splitmix64(base_hashes ^ salt)
        order = np.argsort(keys, kind="stable")
        cumulative = np.cumsum(row_counts[order])
        insertion = int(np.searchsorted(cumulative, target_rows, side="left"))
        candidate_sizes = {
            max(1, min(len(order) - 1, insertion + delta))
            for delta in range(-3, 5)
        }
        for size in sorted(candidate_sizes):
            chosen = order[:size]
            dev_rows = int(row_counts[chosen].sum())
            train_rows = total_rows - dev_rows
            if dev_rows <= 0 or train_rows <= 0:
                continue
            dev_labels_raw = label_counts[chosen].sum(axis=0)
            train_labels_raw = label_counts.sum(axis=0) - dev_labels_raw
            if np.any(dev_labels_raw == 0) or np.any(train_labels_raw == 0):
                continue
            dev_signatures_raw = signature_counts[chosen].sum(axis=0)
            train_signatures_raw = (
                signature_counts.sum(axis=0) - dev_signatures_raw
            )
            dev_strata_raw = stratum_counts[chosen].sum(axis=0)
            train_strata_raw = stratum_counts.sum(axis=0) - dev_strata_raw
            if (
                np.any(dev_signatures_raw[constrained_signatures] == 0)
                or np.any(train_signatures_raw[constrained_signatures] == 0)
                or np.any(dev_strata_raw[constrained_strata] == 0)
                or np.any(train_strata_raw[constrained_strata] == 0)
            ):
                continue

            row_error = abs((dev_rows / total_rows) - fraction)
            label_tv = total_variation(
                dev_labels_raw / dev_rows, global_label
            )
            signature_tv = total_variation(
                dev_signatures_raw / dev_rows,
                global_signature,
            )
            stratum_tv = total_variation(
                dev_strata_raw / dev_rows,
                global_stratum,
            )
            score = (
                25.0 * row_error
                + 4.0 * label_tv
                + 4.0 * signature_tv
                + 2.0 * stratum_tv
            )
            selected_names = [event_ids[index] for index in chosen.tolist()]
            tie_hash = sha256_lines(sorted(selected_names))
            key = (float(score), tie_hash)
            if best_key is None or key < best_key:
                best_key = key
                best_indices = chosen.copy()
                best_metrics = {
                    "total": float(score),
                    "row_fraction_error": float(row_error),
                    "label_total_variation": float(label_tv),
                    "signature_total_variation": float(signature_tv),
                    "label_x_signature_total_variation": float(stratum_tv),
                }

    if best_indices is None or best_metrics is None:
        raise RuntimeError("grouped stratified search found no valid dev split")
    selected_events = {event_ids[index] for index in best_indices.tolist()}
    achieved_rows = int(train["event_id"].isin(selected_events).sum())
    return selected_events, {
        "algorithm": (
            "train-only stable-SHA256 event seeds + SplitMix64 permutation "
            "search; random-prefix candidate closest to target rows; objective "
            "matches label and availability-signature distributions"
        ),
        "objective_weights": {
            "row_fraction_error": 25.0,
            "label_total_variation": 4.0,
            "signature_total_variation": 4.0,
            "label_x_signature_total_variation": 2.0,
        },
        "coverage_constraints": {
            "signature_min_source_events_for_both_arms": 5,
            "label_x_signature_min_source_events_for_both_arms": 10,
        },
        "seed": int(seed),
        "search_trials": int(search_trials),
        "target_fraction": float(fraction),
        "achieved_fraction": float(achieved_rows / len(train)),
        "source_train_events": int(train["event_id"].nunique()),
        "selected_dev_events": int(len(selected_events)),
        "selected_dev_event_ids_sha256": sha256_lines(
            sorted(selected_events)
        ),
        "objective": best_metrics,
        "test_manifest_used_for_selection": False,
    }


def counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    label_signature = (
        frame.groupby(
            ["availability_signature", "label"], observed=True
        )
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for label in (0, 1):
        if label not in label_signature.columns:
            label_signature[label] = 0
    label_signature = label_signature[[0, 1]]
    return {
        "rows": int(len(frame)),
        "labels": counts_dict(frame["label"]),
        "positive_fraction": float(frame["label"].mean()),
        "ids": int(frame["id"].nunique()),
        "plume_ids": int(frame["plume_id"].nunique()),
        "event_ids": int(frame["event_id"].nunique()),
        "cluster_ids": int(frame["cluster_id"].nunique()),
        "macro_region_ids": int(frame["macro_region_id"].nunique()),
        "availability_signatures": counts_dict(
            frame["availability_signature"]
        ),
        "availability_signature_by_label": {
            str(signature): {
                "0": int(row[0]),
                "1": int(row[1]),
            }
            for signature, row in label_signature.iterrows()
        },
        "sensor_available_rows": {
            sensor: int(
                frame["availability_signature"]
                .str.split("+", regex=False)
                .map(lambda values: sensor in values)
                .sum()
            )
            for sensor in SENSORS
        },
        "query360_index_min": int(frame["query360_index"].min()),
        "query360_index_max": int(frame["query360_index"].max()),
    }


def overlap_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in (
        "id",
        "query360_index",
        "plume_id",
        "event_id",
        "cluster_id",
        "macro_region_id",
    ):
        overlap = set(left[column].astype(str)) & set(right[column].astype(str))
        output[column] = {
            "count": int(len(overlap)),
            "examples": sorted(overlap)[:20],
        }
    return output


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        digest = sha256_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_written(
    paths: Mapping[str, Path],
    expected: Mapping[str, pd.DataFrame],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    verification: dict[str, Any] = {}
    usecols = [
        "id",
        "plume_id",
        "event_id",
        "label",
        "query360_index",
        "cluster_id",
        "macro_region_id",
        "availability_signature",
    ]
    for role, path in paths.items():
        observed_hash = sha256_file(path)
        if observed_hash != hashes[role]:
            raise RuntimeError(f"post-write hash mismatch for {role}: {path}")
        observed = pd.read_csv(
            path,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        if len(observed) != len(expected[role]):
            raise RuntimeError(
                f"post-write row mismatch for {role}: "
                f"{len(observed)} != {len(expected[role])}"
            )
        if observed["id"].tolist() != expected[role]["id"].astype(str).tolist():
            raise RuntimeError(f"post-write id/order mismatch for {role}")
        verification[role] = {
            "sha256_recomputed": observed_hash,
            "rows_reloaded": int(len(observed)),
            "id_order_sha256": sha256_lines(observed["id"].tolist()),
        }
    return verification


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.train_csv).expanduser().absolute()
    evaluation_path = Path(args.test_csv).expanduser().absolute()
    metadata_paths = [
        Path(args.metadata_train_csv).expanduser().absolute(),
        Path(args.metadata_test_csv).expanduser().absolute(),
    ]
    output_dir = Path(args.output_dir).expanduser().absolute()
    output_paths = {
        "train_core": output_dir / "legacy360_train_core.csv",
        "dev": output_dir / "legacy360_dev.csv",
        "test": output_dir / "legacy360_test.csv",
    }
    audit_path = output_dir / "legacy360_manifest_audit.json"
    existing = [
        str(path)
        for path in [*output_paths.values(), audit_path]
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing outputs: {existing}"
        )

    train_source = read_source(train_path, "legacy train")
    evaluation_source = read_source(evaluation_path, "legacy test")
    metadata = read_metadata(metadata_paths)
    train, evaluation = enrich_sources(
        train_source, evaluation_source, metadata
    )
    dev_events, selection = choose_dev_events(
        train,
        seed=int(args.seed),
        fraction=float(args.dev_fraction),
        search_trials=int(args.search_trials),
    )
    dev_mask = train["event_id"].isin(dev_events)
    train_core = train.loc[~dev_mask].copy()
    dev = train.loc[dev_mask].copy()
    if train_core.empty or dev.empty:
        raise RuntimeError("train_core or dev is empty")
    if set(train_core["event_id"]) & set(dev["event_id"]):
        raise RuntimeError("train_core/dev canonical event leakage")
    if len(train_core) + len(dev) != len(train):
        raise RuntimeError("train rows were lost during split")
    if set(train_core["id"]) | set(dev["id"]) != set(train["id"]):
        raise RuntimeError("train_core/dev id union differs from source train")

    outputs = {
        "train_core": train_core,
        "dev": dev,
        "test": evaluation,
    }
    output_hashes = {
        role: atomic_write_csv(frame, output_paths[role])
        for role, frame in outputs.items()
    }
    written_verification = verify_written(
        output_paths, outputs, output_hashes
    )

    pairwise = {
        "train_core__dev": overlap_summary(train_core, dev),
        "train_core__test": overlap_summary(train_core, evaluation),
        "dev__test": overlap_summary(dev, evaluation),
        "source_train__source_test": overlap_summary(train, evaluation),
    }
    audit: dict[str, Any] = {
        "schema_version": "legacy360-dual-axis-manifests-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_policy": {
            "image_or_npz_payloads_opened": False,
            "manifest_rows_subsampled": False,
            "dev_selection_inputs": (
                "legacy train rows only: event_id, label, "
                "availability_signature"
            ),
            "s5p_availability_rule": (
                "non-empty .npz s5p_0_path only; s5p_90_path and "
                "s5p_360_path ignored"
            ),
        },
        "inputs": {
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "rows": int(len(train_source)),
            },
            "test": {
                "path": str(evaluation_path),
                "sha256": sha256_file(evaluation_path),
                "rows": int(len(evaluation_source)),
            },
            "metadata": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in metadata_paths
            ],
            "metadata_join": {
                "key": "globally unique id over both metadata files",
                "source_ids": int(len(train_source) + len(evaluation_source)),
                "metadata_ids": int(len(metadata)),
                "id_universe_exact_match": True,
                "_sample_id_matches_source_plume_id": True,
            },
        },
        "query360_index": {
            "rule": (
                "global source-order integer: legacy train first, then "
                "legacy test"
            ),
            "globally_unique": True,
            "range": [0, int(len(train) + len(evaluation) - 1)],
        },
        "event_id": {
            "rule": "strip-terminal-hyphen-alphanumeric-suffix-v1",
            "regex": EVENT_SUFFIX_RE.pattern,
        },
        "dev_selection": selection,
        "splits": {
            role: {
                **summarize(frame),
                "path": str(output_paths[role]),
                "sha256": output_hashes[role],
            }
            for role, frame in outputs.items()
        },
        "overlaps": pairwise,
        "assertions": {
            "train_core_dev_id_overlap": 0,
            "train_core_dev_plume_overlap": 0,
            "train_core_dev_event_overlap": 0,
            "train_core_plus_dev_equals_source_train": True,
            "source_train_test_id_overlap": 0,
            "source_train_test_plume_overlap": int(
                pairwise["source_train__source_test"]["plume_id"]["count"]
            ),
            "source_train_test_event_overlap_observed_not_modified": int(
                pairwise["source_train__source_test"]["event_id"]["count"]
            ),
        },
        "post_write_verification": written_verification,
    }
    atomic_write_json(audit, audit_path)
    print(json.dumps({
        "audit": str(audit_path),
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "rows": {key: len(value) for key, value in outputs.items()},
        "dev_fraction": selection["achieved_fraction"],
        "source_train_test_event_overlap": pairwise[
            "source_train__source_test"
        ]["event_id"]["count"],
    }, indent=2, sort_keys=True))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default=str(DEFAULT_LEGACY_ROOT / "manifest_time_train.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(DEFAULT_LEGACY_ROOT / "manifest_time_test.csv"),
    )
    parser.add_argument(
        "--metadata-train-csv",
        default=str(
            DEFAULT_METADATA_ROOT
            / "manifest_time_train_360m_macroregion_by_plume.csv"
        ),
    )
    parser.add_argument(
        "--metadata-test-csv",
        default=str(
            DEFAULT_METADATA_ROOT
            / "manifest_time_test_360m_macroregion_by_plume.csv"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--search-trials", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
