#!/usr/bin/env python3
"""Strict data utilities for the 360 m TransientQuery experiment.

This module deliberately has a narrow scope:

* derive an inner train/validation split from one explicitly supplied *train*
  manifest, without consulting any external split;
* stage only classification inputs into a content-addressed local cache;
* load the four 360 m sensor families while preserving sensor and temporal axes;
* fail closed on malformed or unreadable rows (there is no random replacement).

The public collation contract is designed for grouped Panopticon micro-batches.
``query360_collate`` returns row metadata plus one observation batch per sensor.
The ``rows`` entries inside those sensor batches are the globally unique
``query360_index`` values written into the derived CSVs, not local batch
positions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.sensor_transforms import (
    DEFAULT_WV3_BANDS,
    L89_PRECOMPUTED_STATS,
    S2_PRECOMPUTED_STATS,
    S5P_PRECOMPUTED_STATS,
    load_wv3_channel_ids_from_srf,
)
from thirdparty.dinov2.utils.data import extract_wavemus, load_ds_cfg


SENSOR_ORDER: tuple[str, ...] = ("s2", "l89", "emit", "s5p")
ROLE_ORDER: tuple[int, ...] = (0, 1, 2)
ROLE_NAMES: tuple[str, ...] = ("0", "90", "360")
BANNED_PATH_SUBSTRINGS: tuple[str, ...] = ("test", "sealed", "holdout")

OPTICAL_PATH_COLUMNS: dict[str, tuple[str, str, str]] = {
    sensor: tuple(f"{sensor}_{role}_path" for role in ("0", "90", "360"))
    for sensor in ("s2", "l89", "emit")
}
S5P_PATH_COLUMN = "s5p_0_path"
CLASSIFICATION_PATH_COLUMNS: tuple[str, ...] = (
    *OPTICAL_PATH_COLUMNS["s2"],
    *OPTICAL_PATH_COLUMNS["l89"],
    *OPTICAL_PATH_COLUMNS["emit"],
    S5P_PATH_COLUMN,
)
REQUIRED_SPLIT_COLUMNS: tuple[str, ...] = (
    "id",
    "plume_id",
    "label",
    "cluster_id",
    "macro_region_id",
    *CLASSIFICATION_PATH_COLUMNS,
)
DEFAULT_WV3_SRF = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "manifests"
    / "WV3_VNIR_SWIR_response.csv"
)


class Query360DataError(RuntimeError):
    """Base exception for strict 360 m data failures."""


class UnsafePathError(Query360DataError, ValueError):
    """Raised when a path could refer to a held-out or test artifact."""


class CacheIntegrityError(Query360DataError):
    """Raised when a cache copy is absent, truncated, or otherwise inconsistent."""


@dataclass(frozen=True)
class SplitArtifacts:
    train_csv: str
    inner_val_csv: str
    audit_json: str
    source_sha256: str
    train_sha256: str
    inner_val_sha256: str
    train_rows: int
    inner_val_rows: int


@dataclass(frozen=True)
class CacheWarmupReport:
    cache_dir: str
    requested_paths: int
    copied_paths: int
    reused_paths: int
    total_source_bytes: int
    max_workers: int


def _path_text(path: os.PathLike[str] | str) -> str:
    text = os.fspath(path).strip()
    if not text:
        raise UnsafePathError("An empty path is not accepted.")
    return text


def assert_safe_path(path: os.PathLike[str] | str, *, purpose: str = "path") -> Path:
    """Reject paths having any component containing test/sealed/holdout.

    The normalized absolute lexical path is checked without resolving
    symlinks.  Avoiding ``Path.resolve`` here is important: computing cache
    destinations must not issue metadata I/O to the remote source filesystem.
    A substring check is intentional: names such as
    ``manifest_time_test.csv`` and ``sealed-v2`` must be rejected even though
    the forbidden word is not a complete component.
    """

    text = _path_text(path)
    lexical = Path(text).expanduser()
    absolute = Path(os.path.abspath(os.fspath(lexical)))
    candidates = (lexical, absolute)
    for candidate in candidates:
        for component in candidate.parts:
            folded = component.casefold()
            token = next(
                (bad for bad in BANNED_PATH_SUBSTRINGS if bad in folded), None
            )
            if token is not None:
                raise UnsafePathError(
                    f"Refusing {purpose} containing forbidden component substring "
                    f"'{token}': {candidate}"
                )
    return absolute


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk_bytes: int = 8 << 20) -> str:
    safe = assert_safe_path(path, purpose="file to hash")
    digest = hashlib.sha256()
    with safe.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise Query360DataError(f"Manifest is missing required columns: {missing}")


def sensor_availability(row: Mapping[str, Any]) -> tuple[bool, bool, bool, bool]:
    """Return strict full-history availability in ``SENSOR_ORDER``."""

    available: list[bool] = []
    for sensor in ("s2", "l89", "emit"):
        present = tuple(bool(_clean_text(row.get(column, ""))) for column in OPTICAL_PATH_COLUMNS[sensor])
        if any(present) and not all(present):
            raise Query360DataError(
                f"Partial {sensor} temporal triplet is not accepted: "
                f"{dict(zip(OPTICAL_PATH_COLUMNS[sensor], present))}"
            )
        available.append(all(present))
    available.append(bool(_clean_text(row.get(S5P_PATH_COLUMN, ""))))
    return tuple(available)  # type: ignore[return-value]


def availability_signature(row: Mapping[str, Any]) -> str:
    names = [
        sensor
        for sensor, present in zip(SENSOR_ORDER, sensor_availability(row))
        if present
    ]
    return "+".join(names) if names else "none"


def _availability_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Vectorized strict availability without materializing row dictionaries."""

    columns: list[np.ndarray] = []
    for sensor in ("s2", "l89", "emit"):
        present = np.stack(
            [
                np.fromiter(
                    (bool(_clean_text(value)) for value in frame[column].tolist()),
                    dtype=np.bool_,
                    count=len(frame),
                )
                for column in OPTICAL_PATH_COLUMNS[sensor]
            ],
            axis=1,
        )
        partial = present.any(axis=1) & ~present.all(axis=1)
        if partial.any():
            examples = np.flatnonzero(partial)[:10].tolist()
            raise Query360DataError(
                f"Partial {sensor} temporal triplet is not accepted at rows {examples}."
            )
        columns.append(present.all(axis=1))
    columns.append(
        np.fromiter(
            (
                bool(_clean_text(value))
                for value in frame[S5P_PATH_COLUMN].tolist()
            ),
            dtype=np.bool_,
            count=len(frame),
        )
    )
    return np.stack(columns, axis=1)


def _signatures_from_matrix(matrix: np.ndarray) -> list[str]:
    return [
        "+".join(
            sensor
            for sensor, present in zip(SENSOR_ORDER, row.tolist())
            if present
        )
        or "none"
        for row in matrix
    ]


def _validate_classification_paths(frame: pd.DataFrame) -> None:
    for column in CLASSIFICATION_PATH_COLUMNS:
        if column not in frame.columns:
            continue
        for raw in frame[column].tolist():
            path = _clean_text(raw)
            if path:
                assert_safe_path(path, purpose=f"classification input in '{column}'")


def collect_classification_paths(frame: pd.DataFrame) -> list[str]:
    """Collect unique non-mask inputs used by the classification experiment."""

    _ensure_columns(frame, CLASSIFICATION_PATH_COLUMNS)
    paths: set[str] = set()
    for column in CLASSIFICATION_PATH_COLUMNS:
        for raw in frame[column].tolist():
            text = _clean_text(raw)
            if not text:
                continue
            safe = assert_safe_path(
                text, purpose=f"classification input in '{column}'"
            )
            paths.add(str(safe))
    return sorted(paths)


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        # Lexical rooting makes the components deterministic.
        keep, move = sorted((root_left, root_right))
        self.parent[move] = keep


def _region_is_zero(value: str) -> bool:
    compact = "".join(ch for ch in value.casefold() if ch.isalnum())
    return compact in {"0", "region0", "macroregion0"}


def _region_components(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    """Join regions sharing a plume, cluster, or classification path.

    Selecting whole connected components guarantees that all of those
    identifiers remain disjoint even if a future manifest violates its stated
    hierarchy.
    """

    regions = sorted({_clean_text(v) for v in frame["macro_region_id"].tolist()})
    if "" in regions:
        raise Query360DataError("macro_region_id contains an empty value.")
    union = _UnionFind(regions)
    linking_columns = ("plume_id", "cluster_id", *CLASSIFICATION_PATH_COLUMNS)
    for column in linking_columns:
        first_region: dict[str, str] = {}
        for region_raw, value_raw in zip(
            frame["macro_region_id"].tolist(), frame[column].tolist()
        ):
            region, value = _clean_text(region_raw), _clean_text(value_raw)
            if not value:
                continue
            if value in first_region:
                union.union(first_region[value], region)
            else:
                first_region[value] = region
    grouped: dict[str, list[str]] = defaultdict(list)
    for region in regions:
        grouped[union.find(region)].append(region)
    return sorted(tuple(sorted(values)) for values in grouped.values())


def _counts_by_value(values: pd.Series) -> dict[str, int]:
    counts = values.astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    availability = _availability_matrix(frame).astype(np.int64, copy=False)
    return {
        "rows": int(len(frame)),
        "labels": _counts_by_value(frame["label"]),
        "plume_ids": int(frame["plume_id"].nunique()),
        "cluster_ids": int(frame["cluster_id"].nunique()),
        "macro_region_ids": int(frame["macro_region_id"].nunique()),
        "availability_signatures": _counts_by_value(frame["availability_signature"]),
        "sensor_available_rows": {
            sensor: int(availability[:, index].sum())
            for index, sensor in enumerate(SENSOR_ORDER)
        },
    }


def _candidate_score(
    val_mask: np.ndarray,
    *,
    target_fraction: float,
    labels: np.ndarray,
    sensor_matrix: np.ndarray,
) -> tuple[float, dict[str, float]]:
    total_rows = len(labels)
    val_rows = int(val_mask.sum())
    train_mask = ~val_mask
    row_error = abs((val_rows / total_rows) - target_fraction)
    global_positive = float(labels.mean())
    val_positive = float(labels[val_mask].mean())
    train_positive = float(labels[train_mask].mean())
    label_error = 0.5 * (
        abs(val_positive - global_positive) + abs(train_positive - global_positive)
    )
    global_sensor = sensor_matrix.mean(axis=0)
    val_sensor = sensor_matrix[val_mask].mean(axis=0)
    train_sensor = sensor_matrix[train_mask].mean(axis=0)
    sensor_error = float(
        0.5
        * (
            np.abs(val_sensor - global_sensor).mean()
            + np.abs(train_sensor - global_sensor).mean()
        )
    )
    # Row fraction dominates; label and sensor terms break close ties.
    total = (8.0 * row_error) + (1.5 * label_error) + sensor_error
    return total, {
        "total": float(total),
        "row_fraction_error": float(row_error),
        "label_prevalence_error": float(label_error),
        "sensor_prevalence_error": float(sensor_error),
    }


def _choose_inner_val_regions(
    selected: pd.DataFrame,
    *,
    seed: int,
    target_fraction: float,
    min_val_regions: int,
    search_trials: int,
) -> tuple[set[str], dict[str, Any]]:
    if not 0.0 < target_fraction < 0.5:
        raise ValueError(f"target_fraction must be in (0, 0.5), got {target_fraction}")
    if min_val_regions < 1:
        raise ValueError("min_val_regions must be positive.")
    components = _region_components(selected)
    held_train = {
        component for component in components if any(_region_is_zero(v) for v in component)
    }
    candidates = [component for component in components if component not in held_train]
    if sum(len(component) for component in candidates) < min_val_regions:
        raise Query360DataError(
            f"Cannot reserve {min_val_regions} validation regions while keeping "
            "region 0 (and its leakage-connected component) in train."
        )

    region_values = selected["macro_region_id"].astype(str)
    component_rows: dict[tuple[str, ...], int] = {}
    for component in candidates:
        component_rows[component] = int(region_values.isin(component).sum())
    eligible_rows = sum(component_rows.values())
    desired_rows = target_fraction * len(selected)
    base_probability = min(0.95, max(0.01, desired_rows / max(1, eligible_rows)))

    labels = selected["label"].to_numpy(dtype=np.int64)
    sensor_matrix = _availability_matrix(selected).astype(np.int64, copy=False)
    best_key: Optional[tuple[float, str]] = None
    best_regions: Optional[set[str]] = None
    best_parts: Optional[dict[str, float]] = None

    # Hashed pseudo-random search is independent of Python/NumPy RNG versions.
    for trial in range(max(1, int(search_trials))):
        phase = (trial % 25) - 12
        probability = min(0.98, max(0.01, base_probability + phase * 0.02))
        ranked: list[tuple[float, tuple[str, ...]]] = []
        chosen_components: list[tuple[str, ...]] = []
        for component in candidates:
            key = f"{seed}\0inner-val\0{trial}\0{'|'.join(component)}"
            uniform = int(_sha256_bytes(key)[:16], 16) / float(16**16)
            ranked.append((uniform, component))
            if uniform < probability:
                chosen_components.append(component)
        chosen_regions = {
            region for component in chosen_components for region in component
        }
        if len(chosen_regions) < min_val_regions:
            for _, component in sorted(ranked):
                chosen_regions.update(component)
                if len(chosen_regions) >= min_val_regions:
                    break
        if len(chosen_regions) >= int(selected["macro_region_id"].nunique()):
            continue

        val_mask = region_values.isin(chosen_regions).to_numpy(dtype=bool)
        if not val_mask.any() or val_mask.all():
            continue
        train_mask = ~val_mask
        # A balanced candidate must expose both classes and every represented
        # sensor to both arms.
        if len(np.unique(labels[val_mask])) < 2 or len(np.unique(labels[train_mask])) < 2:
            continue
        represented = sensor_matrix.sum(axis=0) > 0
        if np.any(sensor_matrix[val_mask].sum(axis=0)[represented] == 0):
            continue
        if np.any(sensor_matrix[train_mask].sum(axis=0)[represented] == 0):
            continue

        score, parts = _candidate_score(
            val_mask,
            target_fraction=target_fraction,
            labels=labels,
            sensor_matrix=sensor_matrix,
        )
        region_key = ",".join(sorted(chosen_regions))
        comparison = (score, region_key)
        if best_key is None or comparison < best_key:
            best_key = comparison
            best_regions = chosen_regions
            best_parts = parts

    if best_regions is None or best_parts is None:
        raise Query360DataError(
            "Deterministic macro-region search found no candidate satisfying "
            "class and sensor coverage constraints."
        )
    return best_regions, {
        "target_fraction": float(target_fraction),
        "achieved_fraction": float(
            region_values.isin(best_regions).sum() / len(selected)
        ),
        "min_val_regions": int(min_val_regions),
        "search_trials": int(search_trials),
        "forced_train_regions": sorted(
            region for component in held_train for region in component
        ),
        "selected_val_regions": sorted(best_regions),
        "objective": best_parts,
    }


def _atomic_write_dataframe(frame: pd.DataFrame, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _atomic_write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _intersection_values(
    train: pd.DataFrame, inner_val: pd.DataFrame, column: str
) -> set[str]:
    left = {_clean_text(value) for value in train[column].tolist()}
    right = {_clean_text(value) for value in inner_val[column].tolist()}
    left.discard("")
    right.discard("")
    return left & right


def _path_intersection(train: pd.DataFrame, inner_val: pd.DataFrame) -> set[str]:
    return set(collect_classification_paths(train)) & set(
        collect_classification_paths(inner_val)
    )


def derive_inner_split(
    train_csv: os.PathLike[str] | str,
    output_train_csv: os.PathLike[str] | str,
    output_inner_val_csv: os.PathLike[str] | str,
    output_audit_json: os.PathLike[str] | str,
    *,
    seed: int = 360,
    target_val_fraction: float = 0.15,
    min_val_regions: int = 8,
    search_trials: int = 8192,
    overwrite: bool = False,
) -> SplitArtifacts:
    """Derive leakage-free train/inner-val CSVs from one train manifest.

    One row is retained per ``(plume_id, label, availability_signature)`` by
    taking the minimum SHA256 of ``seed + id``.  Whole macro-region connected
    components are then assigned to inner validation.  Region zero is always
    retained in train.
    """

    source = assert_safe_path(train_csv, purpose="source train CSV")
    if "train" not in source.name.casefold():
        raise UnsafePathError(
            f"The only accepted source is an explicitly named train CSV: {source}"
        )
    outputs = tuple(
        assert_safe_path(path, purpose="derived split output")
        for path in (output_train_csv, output_inner_val_csv, output_audit_json)
    )
    if source in outputs:
        raise Query360DataError("A derived output must not overwrite its source CSV.")
    if len(set(outputs)) != len(outputs):
        raise Query360DataError("Derived output paths must be distinct.")
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing derived artifacts: {existing}"
        )
    if not source.is_file():
        raise FileNotFoundError(source)

    # This is the sole manifest read in the split derivation.
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
    _ensure_columns(frame, REQUIRED_SPLIT_COLUMNS)
    if len(frame) == 0:
        raise Query360DataError("Source train CSV is empty.")
    _validate_classification_paths(frame)

    labels_numeric = pd.to_numeric(frame["label"], errors="raise")
    if not labels_numeric.isin([0, 1]).all():
        raise Query360DataError("Only binary integer labels 0/1 are accepted.")
    frame["label"] = labels_numeric.astype(np.int64)
    for column in ("id", "plume_id", "cluster_id", "macro_region_id"):
        if frame[column].map(_clean_text).eq("").any():
            raise Query360DataError(f"Column '{column}' contains empty values.")
        frame[column] = frame[column].map(_clean_text)

    frame["query360_index"] = np.arange(len(frame), dtype=np.int64)
    signatures = _signatures_from_matrix(_availability_matrix(frame))
    if any(signature == "none" for signature in signatures):
        raise Query360DataError(
            "At least one source row has no complete classification sensor input."
        )
    frame["availability_signature"] = signatures
    frame["_selection_hash"] = [
        _sha256_bytes(f"{seed}\0{sample_id}") for sample_id in frame["id"].tolist()
    ]
    selected = (
        frame.sort_values(
            ["_selection_hash", "query360_index"], kind="mergesort"
        )
        .drop_duplicates(
            ["plume_id", "label", "availability_signature"], keep="first"
        )
        .sort_values("query360_index", kind="mergesort")
        .reset_index(drop=True)
    )
    duplicate_groups = int(len(frame) - len(selected))

    val_regions, search_audit = _choose_inner_val_regions(
        selected,
        seed=int(seed),
        target_fraction=float(target_val_fraction),
        min_val_regions=int(min_val_regions),
        search_trials=int(search_trials),
    )
    val_mask = selected["macro_region_id"].isin(val_regions)
    derived_columns = [
        column for column in selected.columns if column != "_selection_hash"
    ]
    inner_train = selected.loc[~val_mask, derived_columns].copy()
    inner_val = selected.loc[val_mask, derived_columns].copy()
    if inner_train.empty or inner_val.empty:
        raise Query360DataError("Derived train or inner validation split is empty.")

    overlaps = {
        key: _intersection_values(inner_train, inner_val, key)
        for key in ("plume_id", "cluster_id", "macro_region_id")
    }
    overlaps["classification_path"] = _path_intersection(inner_train, inner_val)
    nonzero = {key: sorted(value)[:10] for key, value in overlaps.items() if value}
    if nonzero:
        raise Query360DataError(
            f"Leakage assertion failed; cross-split identifiers/paths found: {nonzero}"
        )

    source_sha = sha256_file(source)
    train_output, val_output, audit_output = outputs
    train_sha = _atomic_write_dataframe(inner_train, train_output)
    val_sha = _atomic_write_dataframe(inner_val, val_output)
    audit: dict[str, Any] = {
        "schema_version": "query360-inner-split-v1",
        "source": {
            "path": str(source),
            "sha256": source_sha,
            "rows": int(len(frame)),
            "read_policy": "single_explicit_train_manifest_only",
        },
        "selection": {
            "seed": int(seed),
            "rule": "min_sha256(seed + NUL + id) per "
            "(plume_id,label,availability_signature)",
            "selected_rows": int(len(selected)),
            "discarded_duplicate_rows": duplicate_groups,
        },
        "region_search": search_audit,
        "train": {
            **_split_summary(inner_train),
            "path": str(train_output),
            "sha256": train_sha,
        },
        "inner_val": {
            **_split_summary(inner_val),
            "path": str(val_output),
            "sha256": val_sha,
        },
        "leakage_assertions": {
            "plume_id_overlap": 0,
            "cluster_id_overlap": 0,
            "macro_region_id_overlap": 0,
            "classification_path_overlap": 0,
            "forbidden_path_components_rejected": True,
            "external_test_manifest_read": False,
        },
        "classification_path_columns": list(CLASSIFICATION_PATH_COLUMNS),
        "plume_mask_columns_read_or_cached": False,
    }
    _atomic_write_json(audit, audit_output)
    return SplitArtifacts(
        train_csv=str(train_output),
        inner_val_csv=str(val_output),
        audit_json=str(audit_output),
        source_sha256=source_sha,
        train_sha256=train_sha,
        inner_val_sha256=val_sha,
        train_rows=len(inner_train),
        inner_val_rows=len(inner_val),
    )


class StrictHashedFileCache:
    """A fail-closed, size-verified cache keyed by the absolute source path."""

    def __init__(self, cache_dir: os.PathLike[str] | str):
        self.cache_dir = assert_safe_path(cache_dir, purpose="local cache directory")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cached_path(self, source: os.PathLike[str] | str) -> Path:
        safe_source = assert_safe_path(source, purpose="cache source")
        digest = hashlib.sha256(str(safe_source).encode("utf-8")).hexdigest()
        suffix = "".join(safe_source.suffixes[-2:]) if safe_source.suffixes else ""
        # Keep compound suffixes such as .ome.tif, but cap path-name noise.
        if len(suffix) > 24:
            suffix = safe_source.suffix
        return self.cache_dir / digest[:2] / f"{digest}{suffix}"

    @staticmethod
    def _strict_size(path: Path, *, role: str) -> int:
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a regular file: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise CacheIntegrityError(f"{role} is empty: {path}")
        return int(size)

    def ensure_local(
        self, source: os.PathLike[str] | str
    ) -> tuple[str, bool, int]:
        safe_source = assert_safe_path(source, purpose="cache source")
        source_size = self._strict_size(safe_source, role="cache source")
        destination = self.cached_path(safe_source)
        if destination.exists():
            cached_size = self._strict_size(destination, role="cached file")
            if cached_size != source_size:
                raise CacheIntegrityError(
                    f"Cached size mismatch for {safe_source}: source={source_size}, "
                    f"cache={cached_size}, destination={destination}"
                )
            return str(destination), False, source_size

        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(safe_source, temporary)
            copied_size = self._strict_size(temporary, role="temporary cache copy")
            source_size_after = self._strict_size(safe_source, role="cache source")
            if source_size_after != source_size or copied_size != source_size:
                raise CacheIntegrityError(
                    f"Source changed or copy truncated for {safe_source}: "
                    f"before={source_size}, after={source_size_after}, copied={copied_size}"
                )
            os.replace(temporary, destination)
            final_size = self._strict_size(destination, role="cached file")
            if final_size != source_size:
                raise CacheIntegrityError(
                    f"Atomic cache result has wrong size for {safe_source}: "
                    f"expected={source_size}, actual={final_size}"
                )
        finally:
            if temporary.exists():
                temporary.unlink()
        return str(destination), True, source_size

    def require_cached(self, source: os.PathLike[str] | str) -> str:
        """Return a staged path without touching source filesystem metadata.

        Formal runs call ``warm_up`` first, where source and destination sizes
        are checked.  Dataset workers subsequently use this method so repeated
        epochs and parallel encoder conditions issue local-cache I/O only.
        """

        destination = self.cached_path(source)
        self._strict_size(destination, role="required cached file")
        return str(destination)

    def warm_up(
        self, paths: Sequence[os.PathLike[str] | str], *, max_workers: int = 32
    ) -> CacheWarmupReport:
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        unique = sorted(
            {
                str(assert_safe_path(path, purpose="cache warmup source"))
                for path in paths
                if _clean_text(path)
            }
        )
        copied = reused = total_bytes = 0
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            futures = {
                executor.submit(self.ensure_local, path): path for path in unique
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    _, was_copied, source_bytes = future.result()
                    total_bytes += int(source_bytes)
                    copied += int(was_copied)
                    reused += int(not was_copied)
                except Exception as exc:  # collect every failed source, then fail closed
                    failures.append(f"{source}: {type(exc).__name__}: {exc}")
        if failures:
            preview = "\n".join(failures[:20])
            raise CacheIntegrityError(
                f"Cache warmup failed for {len(failures)}/{len(unique)} paths:\n{preview}"
            )
        return CacheWarmupReport(
            cache_dir=str(self.cache_dir),
            requested_paths=len(unique),
            copied_paths=copied,
            reused_paths=reused,
            total_source_bytes=total_bytes,
            max_workers=int(max_workers),
        )


def warm_classification_cache(
    manifest_csv: os.PathLike[str] | str,
    cache_dir: os.PathLike[str] | str,
    *,
    max_workers: int = 32,
) -> CacheWarmupReport:
    """Warm all and only classification paths referenced by one manifest."""

    manifest = assert_safe_path(manifest_csv, purpose="cache manifest")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    frame = pd.read_csv(
        manifest,
        usecols=lambda name: name in CLASSIFICATION_PATH_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    _ensure_columns(frame, CLASSIFICATION_PATH_COLUMNS)
    paths = collect_classification_paths(frame)
    return StrictHashedFileCache(cache_dir).warm_up(
        paths, max_workers=max_workers
    )


def get_sensor_channel_ids(
    wv3_srf_csv: os.PathLike[str] | str = DEFAULT_WV3_SRF,
) -> "OrderedDict[str, torch.Tensor]":
    """Return fixed one-dimensional channel IDs in ``SENSOR_ORDER``."""

    srf = assert_safe_path(wv3_srf_csv, purpose="WV3/EMIT SRF CSV")
    ids: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    ids["s2"] = extract_wavemus(load_ds_cfg("s2_12band")).to(torch.int16)
    ids["l89"] = extract_wavemus(load_ds_cfg("landsat89_7band")).to(torch.int16)
    ids["emit"] = load_wv3_channel_ids_from_srf(
        str(srf), DEFAULT_WV3_BANDS
    ).reshape(-1).to(torch.int16)
    ids["s5p"] = torch.zeros(1, dtype=torch.int16)
    expected = {"s2": 12, "l89": 7, "emit": 16, "s5p": 1}
    for sensor in SENSOR_ORDER:
        if ids[sensor].ndim != 1 or len(ids[sensor]) != expected[sensor]:
            raise Query360DataError(
                f"Unexpected fixed channel IDs for {sensor}: {tuple(ids[sensor].shape)}"
            )
    return ids


def _to_chw(
    array: np.ndarray, *, expected_channels: int, path: Path, sensor: str
) -> np.ndarray:
    if not isinstance(array, np.ndarray) or array.size == 0:
        raise Query360DataError(f"{sensor} returned an empty non-array at {path}")
    if array.ndim == 2:
        if expected_channels != 1:
            raise Query360DataError(
                f"{sensor} expected {expected_channels} channels, got 2-D {array.shape}: {path}"
            )
        array = array[None, ...]
    elif array.ndim == 3:
        first, last = int(array.shape[0]), int(array.shape[-1])
        if first == expected_channels:
            pass
        elif last == expected_channels:
            array = np.transpose(array, (2, 0, 1))
        elif first > expected_channels and first <= 32:
            array = array[:expected_channels]
        elif last > expected_channels and last <= 32:
            array = np.transpose(array, (2, 0, 1))[:expected_channels]
        else:
            raise Query360DataError(
                f"{sensor} cannot infer a {expected_channels}-channel axis from "
                f"shape {array.shape}: {path}"
            )
    else:
        raise Query360DataError(
            f"{sensor} expected a 2-D/3-D raster, got shape {array.shape}: {path}"
        )
    if array.shape[0] < expected_channels:
        raise Query360DataError(
            f"{sensor} expected at least {expected_channels} channels, got "
            f"{array.shape[0]}: {path}"
        )
    return np.ascontiguousarray(array[:expected_channels])


def _pad_image(image: torch.Tensor, multiple: Optional[int]) -> torch.Tensor:
    if multiple is None:
        return image
    if multiple < 1:
        raise ValueError("pad_to_multiple must be positive or None.")
    _, height, width = image.shape
    target_h = int(math.ceil(height / multiple) * multiple)
    target_w = int(math.ceil(width / multiple) * multiple)
    pad_h, pad_w = target_h - height, target_w - width
    return F.pad(
        image,
        (
            pad_w // 2,
            pad_w - (pad_w // 2),
            pad_h // 2,
            pad_h - (pad_h // 2),
        ),
        value=0.0,
    )


class Query360Dataset(Dataset):
    """Strict four-sensor, three-role 360 m classification dataset."""

    preserve_declared_roles = False

    def __init__(
        self,
        csv_path: os.PathLike[str] | str,
        *,
        local_cache: Optional[StrictHashedFileCache] = None,
        wv3_srf_csv: os.PathLike[str] | str = DEFAULT_WV3_SRF,
        pad_to_multiple: Optional[int] = 14,
        min_finite_fraction: float = 0.05,
        allow_heldout_manifest: bool = False,
    ):
        if allow_heldout_manifest:
            # Held-out access must be an explicit caller decision.  This
            # exception applies only to the manifest path; every raster/NPZ
            # input still passes through ``assert_safe_path`` in ``_resolve``.
            self.csv_path = Path(
                os.path.abspath(os.fspath(Path(csv_path).expanduser()))
            )
        else:
            self.csv_path = assert_safe_path(
                csv_path, purpose="Query360 manifest"
            )
        if not self.csv_path.is_file():
            raise FileNotFoundError(self.csv_path)
        self.frame = pd.read_csv(
            self.csv_path, dtype=str, keep_default_na=False, low_memory=False
        )
        _ensure_columns(self.frame, REQUIRED_SPLIT_COLUMNS)
        if "query360_index" not in self.frame.columns:
            self.frame["query360_index"] = np.arange(len(self.frame), dtype=np.int64)
        indices = pd.to_numeric(self.frame["query360_index"], errors="raise").astype(
            np.int64
        )
        if indices.duplicated().any():
            raise Query360DataError("query360_index must be unique within a manifest.")
        self.frame["query360_index"] = indices
        labels = pd.to_numeric(self.frame["label"], errors="raise").astype(np.int64)
        if not labels.isin([0, 1]).all():
            raise Query360DataError("Only labels 0 and 1 are accepted.")
        self.frame["label"] = labels
        _validate_classification_paths(self.frame)
        recomputed = _signatures_from_matrix(_availability_matrix(self.frame))
        if any(signature == "none" for signature in recomputed):
            raise Query360DataError("A dataset row has no available sensor.")
        if "availability_signature" in self.frame.columns:
            declared = self.frame["availability_signature"].map(_clean_text).tolist()
            mismatch = [
                index
                for index, (left, right) in enumerate(zip(declared, recomputed))
                if left != right
            ]
            if mismatch:
                raise Query360DataError(
                    f"availability_signature mismatch at rows {mismatch[:10]}"
                )
        self.frame["availability_signature"] = recomputed
        self.local_cache = local_cache
        self.pad_to_multiple = pad_to_multiple
        self.min_finite_fraction = float(min_finite_fraction)
        if not 0.0 <= self.min_finite_fraction <= 1.0:
            raise ValueError("min_finite_fraction must be in [0, 1].")
        self.channel_ids = get_sensor_channel_ids(wv3_srf_csv)
        self._s2_mean = torch.tensor(
            S2_PRECOMPUTED_STATS[0], dtype=torch.float32
        ).view(-1, 1, 1)
        self._s2_std = torch.tensor(
            S2_PRECOMPUTED_STATS[1], dtype=torch.float32
        ).clamp_min(1e-6).view(-1, 1, 1)
        self._l89_mean = torch.tensor(
            L89_PRECOMPUTED_STATS[0], dtype=torch.float32
        ).view(-1, 1, 1)
        self._l89_std = torch.tensor(
            L89_PRECOMPUTED_STATS[1], dtype=torch.float32
        ).clamp_min(1e-6).view(-1, 1, 1)
        self._s5p_mean = torch.tensor(S5P_PRECOMPUTED_STATS[0], dtype=torch.float32)
        self._s5p_std = torch.tensor(
            S5P_PRECOMPUTED_STATS[1], dtype=torch.float32
        ).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, raw_path: Any) -> Path:
        source = assert_safe_path(_clean_text(raw_path), purpose="dataset input")
        if self.local_cache is None:
            if not source.is_file():
                raise FileNotFoundError(source)
            return source
        return Path(self.local_cache.require_cached(source))

    def _finalize(
        self,
        raw: torch.Tensor,
        *,
        sensor: str,
        role: int,
    ) -> tuple[torch.Tensor, float, bool, torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(raw)
        finite_fraction = float(finite.to(torch.float32).mean().item())
        finite_raw = torch.where(finite, raw, torch.zeros_like(raw))
        image = finite_raw
        if sensor == "s2":
            image = (image - self._s2_mean) / self._s2_std
        elif sensor == "l89":
            image = (image - self._l89_mean) / self._l89_std
        elif sensor == "emit":
            finite_values = image[finite]
            if finite_values.numel() and float(finite_values.abs().max().item()) > 100.0:
                image = image / 65535.0
        elif sensor == "s5p":
            image = (image - self._s5p_mean[role]) / self._s5p_std[role]
        else:  # pragma: no cover - protected by callers
            raise AssertionError(sensor)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        # Invalid source pixels map to the neutral normalized value, rather
        # than to "(raw zero - mean) / std".
        image = torch.where(finite, image, torch.zeros_like(image))
        image = image.clamp_(-50.0, 50.0)
        image = _pad_image(image, self.pad_to_multiple)
        return (
            image,
            finite_fraction,
            bool(finite_fraction >= self.min_finite_fraction),
            finite_raw,
            finite,
        )

    def _load_tiff(
        self, raw_path: Any, *, sensor: str, role: int
    ) -> tuple[torch.Tensor, float, bool, torch.Tensor, torch.Tensor]:
        path = self._resolve(raw_path)
        expected = int(self.channel_ids[sensor].numel())
        try:
            array = tifffile.imread(path)
        except Exception as exc:
            raise Query360DataError(
                f"Failed to read {sensor} role={role} TIFF {path}: {exc}"
            ) from exc
        array = _to_chw(
            np.asarray(array),
            expected_channels=expected,
            path=path,
            sensor=sensor,
        )
        raw = torch.from_numpy(array.astype(np.float32, copy=False))
        return self._finalize(raw, sensor=sensor, role=role)

    def _load_s5p(
        self, raw_path: Any
    ) -> list[tuple[torch.Tensor, float, bool, torch.Tensor, torch.Tensor]]:
        path = self._resolve(raw_path)
        if path.suffix.casefold() != ".npz":
            raise Query360DataError(f"S5P input must be an NPZ, got: {path}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "ch4" not in payload:
                    raise Query360DataError(
                        f"S5P NPZ does not contain required 'ch4' array: {path}"
                    )
                array = np.asarray(payload["ch4"])
        except Query360DataError:
            raise
        except Exception as exc:
            raise Query360DataError(f"Failed to read S5P NPZ {path}: {exc}") from exc
        if array.ndim != 3:
            raise Query360DataError(
                f"S5P ch4 must have shape [3,H,W] or [H,W,3], got {array.shape}: {path}"
            )
        if array.shape[0] == 3:
            chw = array
        elif array.shape[-1] == 3:
            chw = np.transpose(array, (2, 0, 1))
        else:
            raise Query360DataError(
                f"S5P ch4 must contain exactly three temporal roles, got {array.shape}: {path}"
            )
        outputs = []
        for role in ROLE_ORDER:
            raw = torch.from_numpy(
                np.ascontiguousarray(chw[role : role + 1]).astype(
                    np.float32, copy=False
                )
            )
            outputs.append(self._finalize(raw, sensor="s5p", role=role))
        return outputs

    @staticmethod
    def _mask_duplicate_roles(
        loaded: Sequence[
            tuple[torch.Tensor, float, bool, torch.Tensor, torch.Tensor]
        ]
    ) -> list[tuple[torch.Tensor, float, bool]]:
        output: list[tuple[torch.Tensor, float, bool]] = []
        prior_unique: list[tuple[torch.Tensor, torch.Tensor]] = []
        t0_valid = bool(loaded[0][2])
        for role, (image, fraction, valid, finite_raw, finite_mask) in enumerate(
            loaded
        ):
            # A sensor without a usable current observation cannot contribute a
            # history-only evidence token.
            is_valid = bool(valid) and t0_valid
            if role > 0 and is_valid:
                if any(
                    finite_raw.shape == prior_raw.shape
                    and torch.equal(finite_raw, prior_raw)
                    and torch.equal(finite_mask, prior_mask)
                    for prior_raw, prior_mask in prior_unique
                ):
                    is_valid = False
            output.append((image, fraction, is_valid))
            if is_valid:
                prior_unique.append((finite_raw, finite_mask))
        return output

    def _declared_role_image(
        self,
        image: torch.Tensor,
        *,
        sensor: str,
        role: int,
        finite_raw: torch.Tensor,
        finite_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the image retained for an optional declared-role side path.

        The strict Query360 path uses only finite, current-anchored roles.
        Legacy checkpoint reproduction sometimes also needs readable roles
        that fail that validity gate.  Subclasses can opt into preserving
        those roles and override this hook when the historical preprocessing
        differs from the strict representation.
        """

        del sensor, role, finite_raw, finite_mask
        return image

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.frame.iloc[int(item)]
        available = sensor_availability(row)
        valid_mask = torch.zeros((len(SENSOR_ORDER), len(ROLE_ORDER)), dtype=torch.bool)
        finite_fraction = torch.zeros(
            (len(SENSOR_ORDER), len(ROLE_ORDER)), dtype=torch.float32
        )
        observations: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict(
            (sensor, []) for sensor in SENSOR_ORDER
        )
        declared_observations: "OrderedDict[str, list[dict[str, Any]]]" = (
            OrderedDict((sensor, []) for sensor in SENSOR_ORDER)
        )
        try:
            for sensor_index, (sensor, present) in enumerate(
                zip(SENSOR_ORDER, available)
            ):
                if not present:
                    continue
                if sensor == "s5p":
                    loaded = self._load_s5p(row[S5P_PATH_COLUMN])
                else:
                    loaded = [
                        self._load_tiff(
                            row[column], sensor=sensor, role=role
                        )
                        for role, column in enumerate(OPTICAL_PATH_COLUMNS[sensor])
                    ]
                if self.preserve_declared_roles:
                    for role, (
                        image,
                        fraction,
                        _valid,
                        finite_raw,
                        finite_mask,
                    ) in enumerate(loaded):
                        declared_observations[sensor].append(
                            {
                                "image": self._declared_role_image(
                                    image,
                                    sensor=sensor,
                                    role=role,
                                    finite_raw=finite_raw,
                                    finite_mask=finite_mask,
                                ),
                                "role": int(role),
                                "finite_fraction": float(fraction),
                            }
                        )
                loaded = self._mask_duplicate_roles(loaded)
                for role, (image, fraction, valid) in enumerate(loaded):
                    finite_fraction[sensor_index, role] = float(fraction)
                    valid_mask[sensor_index, role] = bool(valid)
                    if valid:
                        observations[sensor].append(
                            {"image": image, "role": int(role)}
                        )
        except Exception as exc:
            row_index = int(row["query360_index"])
            sample_id = _clean_text(row["id"])
            if isinstance(exc, Query360DataError):
                raise Query360DataError(
                    f"Strict row load failed at query360_index={row_index}, "
                    f"id={sample_id}: {exc}"
                ) from exc
            raise

        return {
            "index": int(row["query360_index"]),
            "label": int(row["label"]),
            "id": _clean_text(row["id"]),
            "plume_id": _clean_text(row["plume_id"]),
            "cluster_id": _clean_text(row["cluster_id"]),
            "macro_region_id": _clean_text(row["macro_region_id"]),
            "availability_signature": _clean_text(
                row["availability_signature"]
            ),
            "observations": observations,
            "valid_mask": valid_mask,
            "finite_fraction": finite_fraction,
            "channel_ids": OrderedDict(
                (sensor, values.clone()) for sensor, values in self.channel_ids.items()
            ),
            **(
                {"declared_observations": declared_observations}
                if self.preserve_declared_roles
                else {}
            ),
        }


def _stack_images(images: Sequence[torch.Tensor]) -> torch.Tensor:
    if not images:
        raise ValueError("_stack_images requires at least one image.")
    channels = int(images[0].shape[0])
    if any(image.ndim != 3 or int(image.shape[0]) != channels for image in images):
        raise Query360DataError("A sensor batch has inconsistent channel counts.")
    max_h = max(int(image.shape[1]) for image in images)
    max_w = max(int(image.shape[2]) for image in images)
    padded = []
    for image in images:
        pad_h, pad_w = max_h - int(image.shape[1]), max_w - int(image.shape[2])
        padded.append(F.pad(image, (0, pad_w, 0, pad_h), value=0.0))
    return torch.stack(padded, dim=0)


def query360_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate rows without collapsing either the sensor or temporal axis.

    Returns
    -------
    dict
        ``index[N]``, ``labels[N]``, metadata lists, ``valid_mask[N,4,3]``,
        ``finite_fraction[N,4,3]`` and an insertion-ordered ``sensor_batches``.
        Every sensor batch contains ``images[M,C,H,W]``, global ``rows[M]``,
        ``roles[M]`` (0/1/2), and fixed ``channel_ids[C]``.
    """

    if not batch:
        raise ValueError("Cannot collate an empty Query360 batch.")
    indices = [int(item["index"]) for item in batch]
    if len(indices) != len(set(indices)):
        raise Query360DataError("A batch contains duplicate query360_index values.")
    sensor_batches: "OrderedDict[str, dict[str, torch.Tensor]]" = OrderedDict()
    preserve_declared_roles = any(
        "declared_observations" in item for item in batch
    )
    if preserve_declared_roles and not all(
        "declared_observations" in item for item in batch
    ):
        raise Query360DataError(
            "A batch mixes declared-role-preserving and strict-only samples."
        )
    declared_sensor_batches: "OrderedDict[str, dict[str, torch.Tensor]]" = (
        OrderedDict()
    )
    first_ids = batch[0]["channel_ids"]
    for sensor in SENSOR_ORDER:
        channel_ids = torch.as_tensor(first_ids[sensor]).reshape(-1).to(torch.int16)
        images: list[torch.Tensor] = []
        rows: list[int] = []
        roles: list[int] = []
        for item in batch:
            current_ids = (
                torch.as_tensor(item["channel_ids"][sensor])
                .reshape(-1)
                .to(torch.int16)
            )
            if not torch.equal(current_ids, channel_ids):
                raise Query360DataError(
                    f"Inconsistent fixed channel IDs within {sensor} batch."
                )
            for observation in item["observations"][sensor]:
                images.append(torch.as_tensor(observation["image"]))
                rows.append(int(item["index"]))
                roles.append(int(observation["role"]))
        if images:
            image_batch = _stack_images(images)
        else:
            image_batch = torch.empty(
                (0, int(channel_ids.numel()), 0, 0), dtype=torch.float32
            )
        sensor_batches[sensor] = {
            "images": image_batch,
            "rows": torch.tensor(rows, dtype=torch.long),
            "roles": torch.tensor(roles, dtype=torch.long),
            "channel_ids": channel_ids,
        }
        if preserve_declared_roles:
            declared_images: list[torch.Tensor] = []
            declared_rows: list[int] = []
            declared_roles: list[int] = []
            for item in batch:
                for observation in item["declared_observations"][sensor]:
                    declared_images.append(torch.as_tensor(observation["image"]))
                    declared_rows.append(int(item["index"]))
                    declared_roles.append(int(observation["role"]))
            if declared_images:
                declared_image_batch = _stack_images(declared_images)
            else:
                declared_image_batch = torch.empty(
                    (0, int(channel_ids.numel()), 0, 0), dtype=torch.float32
                )
            declared_sensor_batches[sensor] = {
                "images": declared_image_batch,
                "rows": torch.tensor(declared_rows, dtype=torch.long),
                "roles": torch.tensor(declared_roles, dtype=torch.long),
                "channel_ids": channel_ids,
            }

    output = {
        "index": torch.tensor(indices, dtype=torch.long),
        "labels": torch.tensor(
            [int(item["label"]) for item in batch], dtype=torch.long
        ),
        "ids": [str(item["id"]) for item in batch],
        "plume_ids": [str(item["plume_id"]) for item in batch],
        "cluster_ids": [str(item["cluster_id"]) for item in batch],
        "macro_region_ids": [str(item["macro_region_id"]) for item in batch],
        "availability_signatures": [
            str(item["availability_signature"]) for item in batch
        ],
        "sensor_batches": sensor_batches,
        "valid_mask": torch.stack(
            [torch.as_tensor(item["valid_mask"], dtype=torch.bool) for item in batch]
        ),
        "finite_fraction": torch.stack(
            [
                torch.as_tensor(item["finite_fraction"], dtype=torch.float32)
                for item in batch
            ]
        ),
    }
    if preserve_declared_roles:
        output["declared_sensor_batches"] = declared_sensor_batches
        # Recreate the historical wide-row collate contract for the frozen
        # checkpoint base: concatenate the three roles into channels first,
        # then pad C/H/W globally across every sensor sample in the minibatch.
        # The old classifier sliced this one globally padded tensor by sensor
        # only after collation.
        concat_images: list[torch.Tensor] = []
        concat_channel_ids: list[torch.Tensor] = []
        concat_rows: list[int] = []
        concat_sensors: list[str] = []
        historical_sensor_order = ("s2", "l89", "s5p", "emit")
        for item in batch:
            for sensor in historical_sensor_order:
                observations = item["declared_observations"][sensor]
                if not observations:
                    continue
                by_role = {
                    int(observation["role"]): torch.as_tensor(
                        observation["image"]
                    )
                    for observation in observations
                }
                if set(by_role) != set(ROLE_ORDER):
                    raise Query360DataError(
                        f"Declared {sensor} roles for query360_index="
                        f"{int(item['index'])} are {sorted(by_role)}, expected "
                        f"{list(ROLE_ORDER)}."
                    )
                concat_images.append(
                    torch.cat([by_role[role] for role in ROLE_ORDER], dim=0)
                )
                concat_channel_ids.append(
                    torch.as_tensor(item["channel_ids"][sensor])
                    .reshape(-1)
                    .repeat(len(ROLE_ORDER))
                    .to(torch.int16)
                )
                concat_rows.append(int(item["index"]))
                concat_sensors.append(sensor)
        if not concat_images:
            raise Query360DataError(
                "Declared-role-preserving batch has no declared sensor."
            )
        max_channels = max(int(image.shape[0]) for image in concat_images)
        max_h = max(int(image.shape[1]) for image in concat_images)
        max_w = max(int(image.shape[2]) for image in concat_images)
        padded_images: list[torch.Tensor] = []
        padded_ids: list[torch.Tensor] = []
        for image, channel_ids in zip(concat_images, concat_channel_ids):
            channels, height, width = map(int, image.shape)
            image = F.pad(
                image, (0, max_w - width, 0, max_h - height), value=0.0
            )
            if channels < max_channels:
                image = torch.cat(
                    [
                        image,
                        image.new_zeros(
                            max_channels - channels, max_h, max_w
                        ),
                    ],
                    dim=0,
                )
                channel_ids = torch.cat(
                    [
                        channel_ids,
                        channel_ids.new_zeros(max_channels - channels),
                    ],
                    dim=0,
                )
            padded_images.append(image)
            padded_ids.append(channel_ids)
        output["legacy_concat_batch"] = {
            "images": torch.stack(padded_images, dim=0),
            "channel_ids": torch.stack(padded_ids, dim=0),
            "rows": torch.tensor(concat_rows, dtype=torch.long),
            "sensors": concat_sensors,
        }
    return output


__all__ = [
    "BANNED_PATH_SUBSTRINGS",
    "CLASSIFICATION_PATH_COLUMNS",
    "CacheIntegrityError",
    "CacheWarmupReport",
    "DEFAULT_WV3_SRF",
    "OPTICAL_PATH_COLUMNS",
    "Query360DataError",
    "Query360Dataset",
    "ROLE_NAMES",
    "ROLE_ORDER",
    "SENSOR_ORDER",
    "S5P_PATH_COLUMN",
    "SplitArtifacts",
    "StrictHashedFileCache",
    "UnsafePathError",
    "assert_safe_path",
    "availability_signature",
    "collect_classification_paths",
    "derive_inner_split",
    "get_sensor_channel_ids",
    "query360_collate",
    "sensor_availability",
    "sha256_file",
    "warm_classification_cache",
]
