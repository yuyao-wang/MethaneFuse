#!/usr/bin/env python3
"""Dev-only low-dimensional AxisStats control for legacy360 caches.

This is deliberately a strong simple engineering control, not an RCTP method.
It consumes only a train-core feature cache and a canonical development cache,
derives scalar time/sensor statistics, trains a small scikit-learn grid, and
locks the development-selected model and threshold.  There is intentionally no
test/evaluation subcommand.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SCHEMA_VERSION = "legacy360-axisstats-control-v1"
FEATURE_SCHEMA_VERSION = "legacy360-axisstats-features-v1"
CACHE_SCHEMA_VERSION = "query360-two-axis-feature-cache-v1"
SENSORS = ("s2", "l89", "emit", "s5p")
ROLES = ("t0", "t90", "t360")
MODES = ("universal", "hybrid")
FORBIDDEN_INPUT_TOKENS = frozenset({"test", "sealed"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.expanduser().absolute().parts:
        tokens.update(re.findall(r"[a-z0-9]+", part.lower()))
    return tokens


def assert_dev_only_input_path(path: Path, role: str) -> None:
    overlap = _path_tokens(path) & FORBIDDEN_INPUT_TOKENS
    if overlap:
        raise ValueError(
            f"{role} input path contains forbidden dev-only token(s) "
            f"{sorted(overlap)}: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"{role} input cache does not exist: {path}")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


@dataclass
class AxisStatsDataset:
    matrix: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    ids: list[str]
    plume_ids: list[str]
    event_ids: list[str]
    availability_signatures: list[str]
    query360_indices: list[int]
    split: str
    cache_path: str
    cache_sha256: str
    manifest: Mapping[str, Any]
    rows: int


class ScalarColumns:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.columns: dict[str, np.ndarray] = {}

    def add(
        self,
        name: str,
        value: np.ndarray | torch.Tensor | Sequence[float],
    ) -> None:
        if name in self.columns:
            raise ValueError(f"duplicate AxisStats feature: {name}")
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        array = np.asarray(array, dtype=np.float32).reshape(-1)
        if len(array) != self.rows:
            raise ValueError(
                f"{name}: rows={len(array)}, expected={self.rows}"
            )
        array = np.nan_to_num(
            array, nan=0.0, posinf=1e6, neginf=-1e6
        ).astype(np.float32, copy=False)
        self.columns[name] = array

    def finish(self) -> tuple[np.ndarray, list[str]]:
        names = list(self.columns)
        if not names:
            raise ValueError("no AxisStats features were constructed")
        matrix = np.column_stack([self.columns[name] for name in names])
        if not np.isfinite(matrix).all():
            raise ValueError("AxisStats matrix contains non-finite values")
        return matrix.astype(np.float32, copy=False), names


def _masked_output(
    value: torch.Tensor, valid: torch.Tensor
) -> np.ndarray:
    array = value.detach().cpu().numpy().astype(np.float32, copy=False)
    mask = valid.detach().cpu().numpy().astype(bool, copy=False)
    return np.where(mask, array, 0.0).astype(np.float32, copy=False)


def _rms_norm(vectors: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(vectors * vectors, dim=-1).clamp_min(0.0))


def _pair_stats(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    denominator = (left_norm * right_norm).clamp_min(1e-12)
    cosine = torch.sum(left * right, dim=-1) / denominator
    l2_rms = torch.sqrt(
        torch.mean((left - right) ** 2, dim=-1).clamp_min(0.0)
    )
    return cosine, l2_rms


def _cache_tensor(
    payload: Mapping[str, Any], key: str, fallback: str | None = None
) -> torch.Tensor:
    if key in payload:
        value = payload[key]
    elif fallback is not None and fallback in payload:
        value = payload[fallback]
    else:
        raise ValueError(f"cache is missing tensor {key}")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"cache field {key} is not a tensor")
    return value


def _string_list(
    payload: Mapping[str, Any], key: str, rows: int
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"cache field {key} is not a row list")
    output = [str(item) for item in value]
    if len(output) != rows:
        raise ValueError(f"cache field {key} has wrong row count")
    return output


def extract_axisstats(
    payload: Mapping[str, Any],
    *,
    cache_path: Path,
    cache_sha256: str,
) -> AxisStatsDataset:
    features_reference = _cache_tensor(payload, "features")
    valid = _cache_tensor(payload, "valid_mask").bool()
    labels = _cache_tensor(payload, "labels").long()
    if features_reference.ndim != 4:
        raise ValueError("features must have [row,sensor,time,dim] shape")
    rows, sensors, roles, dimensions = features_reference.shape
    if (sensors, roles) != (len(SENSORS), len(ROLES)):
        raise ValueError(
            f"unexpected sensor/time shape {(sensors, roles)}"
        )
    if dimensions <= 0 or valid.shape != (rows, sensors, roles):
        raise ValueError("malformed feature or validity tensor")
    if labels.shape != (rows,):
        raise ValueError("malformed labels")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("cache labels must contain both binary classes")
    sensor_names = tuple(str(value) for value in payload.get("sensor_names", []))
    if sensor_names and sensor_names != SENSORS:
        raise ValueError(f"unexpected sensor order: {sensor_names}")

    columns = ScalarColumns(rows)

    # Availability is a first-class control so zeros for missing statistics
    # cannot be mistaken for genuine zero-valued evidence.
    for sensor_index, sensor in enumerate(SENSORS):
        for role_index, role in enumerate(ROLES):
            columns.add(
                f"valid_obs_{sensor}_{role}",
                valid[:, sensor_index, role_index].float(),
            )
    columns.add("count_valid_observations", valid.sum(dim=(1, 2)).float())
    columns.add("count_valid_current_sensors", valid[:, :, 0].sum(dim=1).float())
    columns.add(
        "count_valid_history_observations", valid[:, :, 1:].sum(dim=(1, 2)).float()
    )

    # Exact old fused and per-sensor decision logits, for both engineering
    # bases.  This makes AxisStats a strong control rather than forcing it to
    # relearn an already-good decision boundary from geometry alone.
    for mode in MODES:
        fused = _cache_tensor(
            payload,
            f"base_{mode}_logits",
            "base_fused_logits" if mode == "hybrid" else None,
        ).float()
        sensor_logits = _cache_tensor(
            payload,
            f"base_sensor_logits_{mode}",
            "base_sensor_logits" if mode == "hybrid" else None,
        ).float()
        sensor_valid = _cache_tensor(
            payload,
            f"base_sensor_valid_{mode}",
            "base_sensor_valid" if mode == "hybrid" else None,
        ).bool()
        if fused.shape != (rows,) or sensor_logits.shape != (rows, len(SENSORS)):
            raise ValueError(f"malformed {mode} base logits")
        if sensor_valid.shape != sensor_logits.shape:
            raise ValueError(f"malformed {mode} base sensor validity")
        columns.add(f"base_{mode}_fused_logit", fused)
        for sensor_index, sensor in enumerate(SENSORS):
            columns.add(
                f"base_{mode}_{sensor}_logit",
                torch.where(
                    sensor_valid[:, sensor_index],
                    sensor_logits[:, sensor_index],
                    torch.zeros_like(sensor_logits[:, sensor_index]),
                ),
            )
            columns.add(
                f"valid_base_{mode}_{sensor}",
                sensor_valid[:, sensor_index].float(),
            )

    universal_fused = _cache_tensor(
        payload, "base_universal_logits"
    ).float()
    hybrid_fused = _cache_tensor(
        payload, "base_hybrid_logits", "base_fused_logits"
    ).float()
    columns.add("base_hybrid_minus_universal_logit", hybrid_fused - universal_fused)

    for mode in MODES:
        mode_features = _cache_tensor(
            payload,
            f"features_{mode}",
            "features" if mode == "hybrid" else None,
        )
        if mode_features.shape != features_reference.shape:
            raise ValueError(f"malformed features_{mode}")

        # Keeping one float32 current tensor per mode bounds peak memory while
        # avoiding unstable float16 dot products.
        current = mode_features[:, :, 0, :].float()
        for sensor_index, sensor in enumerate(SENSORS):
            t0 = current[:, sensor_index, :]
            t0_valid = valid[:, sensor_index, 0]
            columns.add(
                f"axis_{mode}_{sensor}_t0_norm_rms",
                _masked_output(_rms_norm(t0), t0_valid),
            )
            for role_index, role in enumerate(ROLES[1:], start=1):
                history = mode_features[:, sensor_index, role_index, :].float()
                pair_valid = t0_valid & valid[:, sensor_index, role_index]
                cosine, l2_rms = _pair_stats(t0, history)
                columns.add(
                    f"axis_{mode}_{sensor}_t0_vs_{role}_cosine",
                    _masked_output(cosine, pair_valid),
                )
                columns.add(
                    f"axis_{mode}_{sensor}_t0_vs_{role}_l2_rms",
                    _masked_output(l2_rms, pair_valid),
                )
                columns.add(
                    f"axis_{mode}_{sensor}_{role}_norm_rms",
                    _masked_output(
                        _rms_norm(history),
                        valid[:, sensor_index, role_index],
                    ),
                )
                columns.add(
                    f"valid_pair_{mode}_{sensor}_t0_{role}",
                    pair_valid.float(),
                )
                del history, cosine, l2_rms

        # Sensor-axis relationship at the current acquisition.  Individual
        # t0-valid bits and explicit pair-valid bits remain in the feature set.
        for left_index, left_sensor in enumerate(SENSORS):
            for right_index in range(left_index + 1, len(SENSORS)):
                right_sensor = SENSORS[right_index]
                pair_valid = valid[:, left_index, 0] & valid[:, right_index, 0]
                cosine, _ = _pair_stats(
                    current[:, left_index, :],
                    current[:, right_index, :],
                )
                columns.add(
                    (
                        f"axis_{mode}_current_{left_sensor}_"
                        f"{right_sensor}_cosine"
                    ),
                    _masked_output(cosine, pair_valid),
                )
                columns.add(
                    (
                        f"valid_current_pair_{mode}_{left_sensor}_"
                        f"{right_sensor}"
                    ),
                    pair_valid.float(),
                )
                del cosine
        del current

    matrix, feature_names = columns.finish()
    del features_reference

    split = str(payload.get("split", ""))
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("cache is missing manifest metadata")
    ids = _string_list(payload, "ids", rows)
    query_indices_raw = payload.get("query360_indices")
    if isinstance(query_indices_raw, Sequence):
        query_indices = [int(value) for value in query_indices_raw]
    else:
        # Some directly extracted v1 caches predate the merged-cache audit
        # field.  Their immutable manifest remains part of the cache metadata,
        # so recover only this audit index after verifying exact ID order.
        manifest_path = Path(str(manifest.get("path", "")))
        assert_dev_only_input_path(manifest_path, "cache manifest")
        manifest_frame = pd.read_csv(
            manifest_path,
            usecols=lambda name: name in {"id", "query360_index"},
        )
        required_columns = {"id", "query360_index"}
        if set(manifest_frame.columns) != required_columns:
            raise ValueError(
                "cache has no query360_indices and its manifest cannot "
                "supply id/query360_index"
            )
        manifest_ids = manifest_frame["id"].astype(str).tolist()
        if manifest_ids != ids:
            raise ValueError(
                "cache has no query360_indices and manifest ID order differs"
            )
        query_indices = (
            manifest_frame["query360_index"].astype(int).tolist()
        )
    if len(query_indices) != rows:
        raise ValueError("query360_indices has wrong row count")

    return AxisStatsDataset(
        matrix=matrix,
        labels=labels.numpy().astype(np.int64, copy=False),
        feature_names=feature_names,
        ids=ids,
        plume_ids=_string_list(payload, "plume_ids", rows),
        event_ids=_string_list(payload, "event_ids", rows),
        availability_signatures=_string_list(
            payload, "availability_signatures", rows
        ),
        query360_indices=query_indices,
        split=split,
        cache_path=str(cache_path),
        cache_sha256=cache_sha256,
        manifest=dict(manifest),
        rows=rows,
    )


def load_axisstats_cache(path: Path, expected_split: str) -> AxisStatsDataset:
    assert_dev_only_input_path(path, expected_split)
    cache_sha256 = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: cache payload is not a mapping")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported feature cache schema")
    if payload.get("split") != expected_split:
        raise ValueError(
            f"{path}: split={payload.get('split')}, expected={expected_split}"
        )
    if bool(payload.get("sealed_test_read", False)):
        raise ValueError(f"{path}: cache says sealed_test_read=true")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping) or "path" not in manifest:
        raise ValueError(f"{path}: missing source manifest metadata")
    manifest_path = Path(str(manifest["path"]))
    manifest_tokens = _path_tokens(manifest_path)
    forbidden = manifest_tokens & FORBIDDEN_INPUT_TOKENS
    if forbidden:
        raise ValueError(
            f"{path}: source manifest contains forbidden token(s) "
            f"{sorted(forbidden)}"
        )
    dataset = extract_axisstats(
        payload, cache_path=path, cache_sha256=cache_sha256
    )
    del payload
    gc.collect()
    return dataset


def best_positive_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_y = labels[order]
    sorted_p = probabilities[order]
    true_positive = np.cumsum(sorted_y == 1)
    false_positive = np.cumsum(sorted_y == 0)
    ends = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
    positives = int(np.sum(labels == 1))
    tp = true_positive[ends].astype(np.float64)
    fp = false_positive[ends].astype(np.float64)
    fn = positives - tp
    score = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2 * tp + fp + fn) > 0,
    )
    best = int(np.argmax(score))
    return float(sorted_p[ends[best]]), float(score[best])


def probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    threshold, best_f1 = best_positive_f1_threshold(labels, probabilities)
    fixed_prediction = (probabilities >= 0.5).astype(np.int64)
    best_prediction = (probabilities >= threshold).astype(np.int64)
    return {
        "rows": int(len(labels)),
        "positives": int(np.sum(labels == 1)),
        "fixed_binary_f1": float(
            f1_score(labels, fixed_prediction, pos_label=1, zero_division=0)
        ),
        "fixed_macro_f1": float(
            f1_score(
                labels,
                fixed_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "fixed_balanced_accuracy": float(
            balanced_accuracy_score(labels, fixed_prediction)
        ),
        "fixed_predicted_positive_rate": float(fixed_prediction.mean()),
        "best_binary_f1": float(best_f1),
        "best_threshold": float(threshold),
        "best_macro_f1": float(
            f1_score(
                labels,
                best_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "best_predicted_positive_rate": float(best_prediction.mean()),
        "ap": float(average_precision_score(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)),
    }


def fixed_threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    prediction = (probabilities >= float(threshold)).astype(np.int64)
    return {
        "rows": int(len(labels)),
        "positives": int(np.sum(labels == 1)),
        "binary_f1": float(
            f1_score(labels, prediction, pos_label=1, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction)
        ),
        "predicted_positive_rate": float(prediction.mean()),
        "ap": float(average_precision_score(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "threshold": float(threshold),
    }


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    feature_set: str
    parameters: Mapping[str, Any]


def candidate_specs(grid: str) -> list[CandidateSpec]:
    if grid == "smoke":
        return [
            CandidateSpec(
                name="logreg_full_c0p3",
                family="logistic_regression",
                feature_set="full",
                parameters={"C": 0.3},
            ),
            CandidateSpec(
                name="histgb_full_lr0p1_leaf7",
                family="hist_gradient_boosting",
                feature_set="full",
                parameters={"learning_rate": 0.1, "max_leaf_nodes": 7},
            ),
        ]
    if grid != "small":
        raise ValueError(f"unsupported grid {grid}")
    output: list[CandidateSpec] = [
        CandidateSpec(
            name="logreg_universal_fused_c0p03",
            family="logistic_regression",
            feature_set="universal_fused",
            parameters={"C": 0.03},
        )
    ]
    for feature_set in ("fused", "logits", "base"):
        for c_value, c_name in ((0.03, "0p03"), (0.3, "0p3")):
            output.append(
                CandidateSpec(
                    name=f"logreg_{feature_set}_c{c_name}",
                    family="logistic_regression",
                    feature_set=feature_set,
                    parameters={"C": c_value},
                )
            )
    output.append(
        CandidateSpec(
            name="logreg_full_c0p03",
            family="logistic_regression",
            feature_set="full",
            parameters={"C": 0.03},
        )
    )
    for feature_set, learning_rate, rate_name, leaves in (
        ("logits", 0.05, "0p05", 7),
        ("base", 0.05, "0p05", 7),
        ("base", 0.1, "0p1", 15),
        ("full", 0.05, "0p05", 7),
        ("full", 0.1, "0p1", 15),
    ):
        output.append(
            CandidateSpec(
                name=f"histgb_{feature_set}_lr{rate_name}_leaf{leaves}",
                family="hist_gradient_boosting",
                feature_set=feature_set,
                parameters={
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": leaves,
                },
            )
        )
    return output


def base_feature_indices(feature_names: Sequence[str]) -> list[int]:
    prefixes = ("base_", "valid_", "count_")
    indices = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith(prefixes)
    ]
    if not indices:
        raise ValueError("base feature set is empty")
    return indices


def feature_indices_for_set(
    feature_names: Sequence[str], feature_set: str
) -> list[int]:
    if feature_set == "universal_fused":
        expected = "base_universal_fused_logit"
        if expected not in feature_names:
            raise ValueError(f"missing feature {expected}")
        return [list(feature_names).index(expected)]
    if feature_set == "fused":
        expected = {
            "base_universal_fused_logit",
            "base_hybrid_fused_logit",
            "base_hybrid_minus_universal_logit",
        }
        indices = [
            index
            for index, name in enumerate(feature_names)
            if name in expected
        ]
        if {feature_names[index] for index in indices} != expected:
            raise ValueError("fused feature set is incomplete")
        return indices
    if feature_set == "logits":
        indices = [
            index
            for index, name in enumerate(feature_names)
            if name.startswith("base_") and name.endswith("_logit")
        ]
        if not indices:
            raise ValueError("logits feature set is empty")
        return indices
    if feature_set == "base":
        return base_feature_indices(feature_names)
    if feature_set == "full":
        return list(range(len(feature_names)))
    raise ValueError(f"unsupported feature set {feature_set}")


def build_pipeline(
    spec: CandidateSpec,
    *,
    feature_names: Sequence[str],
    seed: int,
    max_iter: int,
) -> tuple[Pipeline, list[int]]:
    indices = feature_indices_for_set(feature_names, spec.feature_set)
    selector = ColumnTransformer(
        [("selected", "passthrough", indices)],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    if spec.family == "logistic_regression":
        classifier = LogisticRegression(
            C=float(spec.parameters["C"]),
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
            tol=1e-4,
        )
        pipeline = Pipeline(
            [
                ("select", selector),
                ("scale", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    elif spec.family == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=True,
            l2_regularization=1.0,
            learning_rate=float(spec.parameters["learning_rate"]),
            max_iter=max_iter,
            max_leaf_nodes=int(spec.parameters["max_leaf_nodes"]),
            min_samples_leaf=40,
            n_iter_no_change=10,
            random_state=seed,
            scoring="loss",
            validation_fraction=0.1,
        )
        pipeline = Pipeline(
            [("select", selector), ("classifier", classifier)]
        )
    else:
        raise ValueError(f"unsupported candidate family {spec.family}")
    return pipeline, indices


def fitted_iterations(model: Pipeline) -> int:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "n_iter_"):
        value = getattr(classifier, "n_iter_")
        return int(np.asarray(value).reshape(-1).max())
    return -1


def split_guard(
    train: AxisStatsDataset, dev: AxisStatsDataset
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for field in ("ids", "plume_ids", "event_ids", "query360_indices"):
        train_values = set(getattr(train, field))
        dev_values = set(getattr(dev, field))
        overlap = train_values & dev_values
        checks[f"{field}_overlap"] = int(len(overlap))
        checks[f"{field}_train_unique"] = int(len(train_values))
        checks[f"{field}_dev_unique"] = int(len(dev_values))
        if overlap:
            examples = sorted(str(value) for value in overlap)[:5]
            raise ValueError(
                f"train/dev {field} overlap={len(overlap)} examples={examples}"
            )
    return checks


def winner_by_sensor(
    dev: AxisStatsDataset,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    signatures = np.asarray(dev.availability_signatures, dtype=object)
    output: dict[str, Any] = {}
    for sensor in SENSORS:
        positions = np.flatnonzero(
            np.asarray(
                [sensor in str(value).split("+") for value in signatures],
                dtype=bool,
            )
        )
        if not len(positions):
            continue
        labels = dev.labels[positions]
        if len(np.unique(labels)) < 2:
            continue
        output[sensor] = fixed_threshold_metrics(
            labels, probabilities[positions], threshold
        )
    return output


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is non-empty; refusing overwrite: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def format_results_markdown(
    records: Sequence[Mapping[str, Any]],
    winner: Mapping[str, Any],
    by_sensor: Mapping[str, Any],
    train: AxisStatsDataset,
    dev: AxisStatsDataset,
) -> str:
    reference = max(
        (record for record in records if record["family"] == "reference"),
        key=lambda record: (
            record["best_binary_f1"],
            record["ap"],
            record["auc"],
        ),
    )
    f1_delta = (
        float(winner["best_binary_f1"])
        - float(reference["best_binary_f1"])
    )
    best_full = max(
        (
            record
            for record in records
            if record["family"] != "reference"
            and record["feature_set"] == "full"
        ),
        key=lambda record: (
            record["best_binary_f1"],
            record["ap"],
            record["auc"],
        ),
    )
    full_f1_delta = (
        float(best_full["best_binary_f1"])
        - float(reference["best_binary_f1"])
    )
    lines = [
        "# Legacy360 AxisStats dev-only control",
        "",
        "This is a strong low-dimensional engineering control, not RCTP "
        "novelty and not a sealed-test result.",
        "",
        f"- Train cache rows: {train.rows:,} (`{train.split}`)",
        f"- Canonical dev rows: {dev.rows:,} (`{dev.split}`)",
        "- Test/sealed inputs: **not accepted and not read**",
        f"- Selected model: `{winner['name']}`",
        f"- Selected dev threshold: `{winner['best_threshold']:.8f}`",
        f"- Strongest direct-PTH reference: `{reference['name']}` at "
        f"dev-best F1 `{reference['best_binary_f1']:.6f}`",
        f"- Fitted-control delta vs strongest reference: `{f1_delta:+.6f}`",
        f"- Best model using all time×sensor statistics: "
        f"`{best_full['name']}` at F1 `{best_full['best_binary_f1']:.6f}` "
        f"(delta `{full_f1_delta:+.6f}`)",
        "",
        "| Candidate | Family | Features | Iter | F1@0.5 | Dev-best F1 | "
        "Threshold | AP | AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {name} | {family} | {feature_set} | {n_iter} | "
            "{fixed_binary_f1:.6f} | {best_binary_f1:.6f} | "
            "{best_threshold:.6f} | {ap:.6f} | {auc:.6f} |".format(
                **record
            )
        )
    lines.extend(
        [
            "",
            "## Locked winner by sensor-containing stratum",
            "",
            "| Sensor | Rows | F1 | Macro F1 | AP | AUC | Positive rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sensor in SENSORS:
        metrics = by_sensor.get(sensor)
        if metrics is None:
            continue
        lines.append(
            f"| {sensor} | {metrics['rows']} | "
            f"{metrics['binary_f1']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['ap']:.6f} | {metrics['auc']:.6f} | "
            f"{metrics['predicted_positive_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Selection and threshold optimization both used canonical dev. "
            "The lock is suitable for a later authorized one-shot evaluation, "
            "but this script intentionally provides no test command.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_iter <= 0 or args.max_iter > 100:
        raise ValueError("--max-iter must be in [1,100]")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    output_dir = Path(args.output_dir).expanduser().absolute()
    prepare_output_directory(output_dir)

    print(f"[axisstats] loading train cache: {args.train_cache}", flush=True)
    train = load_axisstats_cache(Path(args.train_cache), "train_core")
    print(
        f"[axisstats] train rows={train.rows} features={train.matrix.shape[1]}",
        flush=True,
    )
    print(f"[axisstats] loading canonical dev cache: {args.dev_cache}", flush=True)
    dev = load_axisstats_cache(Path(args.dev_cache), "dev")
    print(
        f"[axisstats] dev rows={dev.rows} features={dev.matrix.shape[1]}",
        flush=True,
    )
    if train.feature_names != dev.feature_names:
        raise ValueError("train/dev AxisStats feature schema differs")
    guard = split_guard(train, dev)

    feature_schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "features": train.feature_names,
        "feature_count": len(train.feature_names),
        "feature_sets": {
            feature_set: feature_indices_for_set(
                train.feature_names, feature_set
            )
            for feature_set in (
                "universal_fused",
                "fused",
                "logits",
                "base",
                "full",
            )
        },
        "definition": {
            "base": (
                "universal/hybrid fused and per-sensor logits plus explicit "
                "observation/base/pair validity counts"
            ),
            "time_axis": (
                "per mode and sensor: t0/history cosine, L2 RMS, and RMS norms"
            ),
            "sensor_axis": (
                "per mode: pairwise current-acquisition sensor cosine"
            ),
            "missing_value": (
                "invalid statistics are zero with explicit validity features"
            ),
        },
    }
    atomic_json(output_dir / "axisstats_feature_schema.json", feature_schema)

    candidate_records: list[dict[str, Any]] = []
    dev_probability_columns: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}

    # Non-trained references make the strength of the old PTH boundaries
    # explicit, but model selection below is restricted to fitted controls.
    name_to_index = {
        name: index for index, name in enumerate(train.feature_names)
    }
    for reference_name, feature_name in (
        ("reference_universal_base", "base_universal_fused_logit"),
        ("reference_hybrid_base", "base_hybrid_fused_logit"),
    ):
        probability = sigmoid(dev.matrix[:, name_to_index[feature_name]])
        metrics = probability_metrics(dev.labels, probability)
        candidate_records.append(
            {
                "name": reference_name,
                "family": "reference",
                "feature_set": "fused_logit",
                "parameters": {},
                "n_iter": 0,
                "eligible_for_model_lock": False,
                **metrics,
            }
        )
        dev_probability_columns[reference_name] = probability

    specs = candidate_specs(args.grid)
    with threadpool_limits(limits=args.threads):
        for ordinal, spec in enumerate(specs, start=1):
            print(
                f"[axisstats] fitting {ordinal}/{len(specs)} {spec.name}",
                flush=True,
            )
            model, indices = build_pipeline(
                spec,
                feature_names=train.feature_names,
                seed=args.seed,
                max_iter=args.max_iter,
            )
            model.fit(train.matrix, train.labels)
            probability = model.predict_proba(dev.matrix)[:, 1]
            metrics = probability_metrics(dev.labels, probability)
            record = {
                "name": spec.name,
                "family": spec.family,
                "feature_set": spec.feature_set,
                "parameters": dict(spec.parameters),
                "selected_feature_count": len(indices),
                "n_iter": fitted_iterations(model),
                "eligible_for_model_lock": True,
                **metrics,
            }
            candidate_records.append(record)
            dev_probability_columns[spec.name] = probability
            fitted_models[spec.name] = model
            print(
                f"[axisstats] {spec.name} fixed_f1="
                f"{metrics['fixed_binary_f1']:.6f} best_f1="
                f"{metrics['best_binary_f1']:.6f} AP={metrics['ap']:.6f} "
                f"AUC={metrics['auc']:.6f}",
                flush=True,
            )

    eligible = [
        record
        for record in candidate_records
        if record["eligible_for_model_lock"]
    ]
    winner = max(
        eligible,
        key=lambda record: (
            record["best_binary_f1"],
            record["ap"],
            record["auc"],
            record["fixed_binary_f1"],
            record["name"],
        ),
    )
    winner_name = str(winner["name"])
    winner_model = fitted_models[winner_name]
    winner_probability = dev_probability_columns[winner_name]
    winner_threshold = float(winner["best_threshold"])
    strongest_reference = max(
        (
            record
            for record in candidate_records
            if record["family"] == "reference"
        ),
        key=lambda record: (
            record["best_binary_f1"],
            record["ap"],
            record["auc"],
        ),
    )
    comparison_to_reference = {
        "reference": strongest_reference["name"],
        "best_binary_f1_delta": float(
            winner["best_binary_f1"]
            - strongest_reference["best_binary_f1"]
        ),
        "ap_delta": float(winner["ap"] - strongest_reference["ap"]),
        "auc_delta": float(winner["auc"] - strongest_reference["auc"]),
        "axisstats_improves_reference": bool(
            winner["best_binary_f1"]
            > strongest_reference["best_binary_f1"] + 1e-12
        ),
    }
    best_full_axisstats = max(
        (
            record
            for record in eligible
            if record["feature_set"] == "full"
        ),
        key=lambda record: (
            record["best_binary_f1"],
            record["ap"],
            record["auc"],
            record["fixed_binary_f1"],
            record["name"],
        ),
    )
    best_full_name = str(best_full_axisstats["name"])
    best_full_probability = dev_probability_columns[best_full_name]
    best_full_threshold = float(best_full_axisstats["best_threshold"])
    full_comparison_to_reference = {
        "reference": strongest_reference["name"],
        "best_binary_f1_delta": float(
            best_full_axisstats["best_binary_f1"]
            - strongest_reference["best_binary_f1"]
        ),
        "ap_delta": float(
            best_full_axisstats["ap"] - strongest_reference["ap"]
        ),
        "auc_delta": float(
            best_full_axisstats["auc"] - strongest_reference["auc"]
        ),
        "full_axisstats_improves_reference": bool(
            best_full_axisstats["best_binary_f1"]
            > strongest_reference["best_binary_f1"] + 1e-12
        ),
    }

    model_payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": train.feature_names,
        "candidate": winner,
        "dev_locked_threshold": winner_threshold,
        "pipeline": winner_model,
        "note": (
            "Dev-selected AxisStats engineering control; not an RCTP model."
        ),
    }
    model_path = output_dir / "axisstats_best_model.joblib"
    atomic_joblib(model_path, model_payload)
    model_sha256 = sha256_file(model_path)
    full_model_path = output_dir / "axisstats_best_full_model.joblib"
    atomic_joblib(
        full_model_path,
        {
            "schema_version": SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": train.feature_names,
            "candidate": best_full_axisstats,
            "dev_locked_threshold": best_full_threshold,
            "pipeline": fitted_models[best_full_name],
            "note": (
                "Best dev-F1 control that uses the complete time×sensor "
                "AxisStats feature set; not an RCTP model."
            ),
        },
    )
    full_model_sha256 = sha256_file(full_model_path)

    records_frame = pd.DataFrame(candidate_records)
    atomic_csv(output_dir / "candidate_metrics.csv", records_frame)
    atomic_json(output_dir / "candidate_metrics.json", candidate_records)

    prediction_frame = pd.DataFrame(
        {
            "id": dev.ids,
            "plume_id": dev.plume_ids,
            "event_id": dev.event_ids,
            "query360_index": dev.query360_indices,
            "availability_signature": dev.availability_signatures,
            "label": dev.labels,
        }
    )
    for name, probability in dev_probability_columns.items():
        prediction_frame[f"prob__{name}"] = probability
    atomic_csv(output_dir / "dev_predictions_all_candidates.csv", prediction_frame)

    winner_prediction_frame = prediction_frame[
        [
            "id",
            "plume_id",
            "event_id",
            "query360_index",
            "availability_signature",
            "label",
        ]
    ].copy()
    winner_prediction_frame["probability"] = winner_probability
    winner_prediction_frame["prediction_at_0_5"] = (
        winner_probability >= 0.5
    ).astype(np.int64)
    winner_prediction_frame["prediction_at_dev_locked_threshold"] = (
        winner_probability >= winner_threshold
    ).astype(np.int64)
    atomic_csv(
        output_dir / "dev_predictions_best.csv", winner_prediction_frame
    )

    by_sensor = winner_by_sensor(
        dev, winner_probability, winner_threshold
    )
    atomic_json(output_dir / "winner_by_sensor.json", by_sensor)
    full_by_sensor = winner_by_sensor(
        dev, best_full_probability, best_full_threshold
    )
    atomic_json(
        output_dir / "best_full_axisstats_by_sensor.json", full_by_sensor
    )

    feature_schema_sha256 = sha256_file(
        output_dir / "axisstats_feature_schema.json"
    )
    candidate_metrics_sha256 = sha256_file(
        output_dir / "candidate_metrics.json"
    )
    predictions_sha256 = sha256_file(
        output_dir / "dev_predictions_best.csv"
    )
    selection_lock = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "selection": {
            "metric": "development best positive-class F1",
            "winner": winner_name,
            "dev_locked_threshold": winner_threshold,
            "winner_metrics": winner,
            "tie_break": [
                "best_binary_f1",
                "ap",
                "auc",
                "fixed_binary_f1",
                "candidate_name",
            ],
            "strongest_direct_pth_reference": strongest_reference,
            "comparison_to_reference": comparison_to_reference,
        },
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
        },
        "best_full_axisstats_model": {
            "path": str(full_model_path),
            "sha256": full_model_sha256,
            "candidate": best_full_axisstats,
            "dev_locked_threshold": best_full_threshold,
            "comparison_to_reference": full_comparison_to_reference,
        },
        "inputs": {
            "train_cache": train.cache_path,
            "train_cache_sha256": train.cache_sha256,
            "train_manifest": dict(train.manifest),
            "dev_cache": dev.cache_path,
            "dev_cache_sha256": dev.cache_sha256,
            "dev_manifest": dict(dev.manifest),
        },
        "artifacts": {
            "feature_schema_sha256": feature_schema_sha256,
            "candidate_metrics_sha256": candidate_metrics_sha256,
            "dev_predictions_best_sha256": predictions_sha256,
        },
        "split_guard": guard,
        "protocol": {
            "train_expected_split": "train_core",
            "dev_expected_split": "dev",
            "test_or_sealed_input_accepted": False,
            "sealed_test_read": False,
            "outer_test_metrics_computed": False,
            "selection_source": "canonical development only",
        },
        "claim_scope": (
            "Strong simple legacy360 engineering control; not RCTP novelty "
            "and not a clean outer-test result."
        ),
    }
    atomic_json(output_dir / "selection_lock.json", selection_lock)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "rows": {"train": train.rows, "dev": dev.rows},
        "features": {
            "full": len(train.feature_names),
            "base": len(base_feature_indices(train.feature_names)),
        },
        "grid": args.grid,
        "max_iter": args.max_iter,
        "threads": args.threads,
        "seed": args.seed,
        "candidate_count": len(candidate_records),
        "fitted_candidate_count": len(eligible),
        "winner": winner,
        "comparison_to_reference": comparison_to_reference,
        "best_full_axisstats": best_full_axisstats,
        "full_axisstats_comparison_to_reference": (
            full_comparison_to_reference
        ),
        "winner_by_sensor": by_sensor,
        "best_full_axisstats_by_sensor": full_by_sensor,
        "reference_metrics": {
            record["name"]: record
            for record in candidate_records
            if record["family"] == "reference"
        },
        "sealed_test_read": False,
        "selection_lock": str(output_dir / "selection_lock.json"),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(
        output_dir / "RESULTS.md",
        format_results_markdown(
            candidate_records, winner, by_sensor, train, dev
        ),
    )
    print(
        f"[axisstats] winner={winner_name} dev_best_f1="
        f"{winner['best_binary_f1']:.6f} threshold={winner_threshold:.6f}",
        flush=True,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a dev-only low-dimensional AxisStats control. Test/sealed "
            "cache inputs are intentionally unsupported."
        )
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid", choices=("smoke", "small"), default="small")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
