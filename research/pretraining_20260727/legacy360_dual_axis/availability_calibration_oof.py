#!/usr/bin/env python3
"""Leakage-controlled, dev-only availability-conditioned calibration.

This script freezes an existing model's dev probabilities and evaluates
threshold-selection rules with grouped five-fold out-of-fold (OOF) calibration.
Every held-out row is classified with thresholds fitted only on the other folds.

It is deliberately *not* a model-training, dual-axis, or pretraining experiment.
The reported differences are post-hoc decision-threshold calibration effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


SENSOR_ORDER = ("s2", "l89", "emit", "s5p")
FORBIDDEN_INPUT_TOKEN = re.compile(r"(^|[_\-.])(test|sealed)([_\-.]|$)", re.I)
STRATEGY_GROUP_COLUMN = {
    "availability_signature": "availability_signature",
    "sensor_count": "sensor_count_key",
    "primary_sensor": "primary_sensor",
    "count_primary": "count_primary_key",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Five-fold, event-grouped OOF threshold calibration on dev only."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Frozen dev prediction CSV with id, label, and probability.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "sanitized_min1024" / "legacy360_dev_sanitized.csv",
        help="Canonical dev manifest used only for strict identity/event metadata join.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "availability_calibration_oof_result.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=root / "AVAILABILITY_CALIBRATION_OOF.md",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=root / "availability_calibration_oof_predictions.csv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--min-group-rows",
        type=int,
        default=128,
        help="Minimum calibration rows before a local threshold is eligible.",
    )
    parser.add_argument(
        "--min-group-class-rows",
        type=int,
        default=16,
        help="Minimum rows from each class before a local threshold is eligible.",
    )
    parser.add_argument(
        "--min-group-events",
        type=int,
        default=8,
        help="Minimum calibration events before a local threshold is eligible.",
    )
    parser.add_argument(
        "--shrink-tau",
        type=float,
        default=256.0,
        help=(
            "Local threshold shrinkage: w=n/(n+tau), then "
            "threshold=(1-w)*global+w*local."
        ),
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_dev_only_input(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} does not exist: {resolved}")
    offending = [part for part in resolved.parts if FORBIDDEN_INPUT_TOKEN.search(part)]
    if offending:
        raise ValueError(
            f"{role} violates dev-only guard; forbidden path component(s): {offending}"
        )
    if "dev" not in resolved.name.lower():
        raise ValueError(
            f"{role} filename must explicitly contain 'dev' under the strict guard: "
            f"{resolved.name}"
        )
    return resolved


def clean_string(series: pd.Series, column: str) -> pd.Series:
    result = series.astype("string").fillna("").str.strip()
    if (result == "").any():
        examples = result.index[result == ""].tolist()[:10]
        raise ValueError(f"Blank {column} values at row indices {examples}")
    return result.astype(str)


def canonical_id(series: pd.Series, column: str) -> pd.Series:
    """Normalize integer-like CSV identifiers without accepting lossy values."""
    values: list[str] = []
    for index, value in series.items():
        if pd.isna(value):
            raise ValueError(f"Missing {column} at row index {index}")
        text = str(value).strip()
        if not text:
            raise ValueError(f"Blank {column} at row index {index}")
        if re.fullmatch(r"[+-]?\d+\.0+", text):
            text = text.split(".", maxsplit=1)[0]
        values.append(text)
    return pd.Series(values, index=series.index, dtype="object")


def validate_signature(value: str) -> str:
    tokens = value.lower().split("+")
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"Malformed availability signature: {value!r}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate sensor in availability signature: {value!r}")
    unknown = sorted(set(tokens).difference(SENSOR_ORDER))
    if unknown:
        raise ValueError(f"Unknown sensor(s) {unknown} in signature {value!r}")
    expected = [sensor for sensor in SENSOR_ORDER if sensor in tokens]
    if tokens != expected:
        raise ValueError(
            f"Non-canonical sensor ordering in {value!r}; expected {'+'.join(expected)!r}"
        )
    return "+".join(tokens)


def read_and_join(
    prediction_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = pd.read_csv(prediction_path)
    manifest = pd.read_csv(manifest_path)
    prediction_required = {
        "id",
        "plume_id",
        "availability_signature",
        "label",
        "probability",
    }
    manifest_required = {
        "id",
        "plume_id",
        "event_id",
        "availability_signature",
        "label",
        "anchor_sensor",
        "query360_index",
    }
    missing_predictions = sorted(prediction_required.difference(predictions.columns))
    missing_manifest = sorted(manifest_required.difference(manifest.columns))
    if missing_predictions:
        raise ValueError(f"Prediction CSV missing columns: {missing_predictions}")
    if missing_manifest:
        raise ValueError(f"Manifest CSV missing columns: {missing_manifest}")

    predictions = predictions.copy()
    manifest = manifest.copy()
    predictions["_join_id"] = canonical_id(predictions["id"], "prediction.id")
    manifest["_join_id"] = canonical_id(manifest["id"], "manifest.id")
    for role, frame in (("predictions", predictions), ("manifest", manifest)):
        duplicates = frame.loc[frame["_join_id"].duplicated(False), "_join_id"].unique()
        if len(duplicates):
            raise ValueError(
                f"{role} has duplicate id values; examples: {duplicates[:10].tolist()}"
            )

    prediction_ids = set(predictions["_join_id"])
    manifest_ids = set(manifest["_join_id"])
    missing_in_manifest = sorted(prediction_ids.difference(manifest_ids))
    missing_in_predictions = sorted(manifest_ids.difference(prediction_ids))
    if missing_in_manifest or missing_in_predictions:
        raise ValueError(
            "Prediction/manifest id sets are not identical: "
            f"prediction-only={missing_in_manifest[:10]}, "
            f"manifest-only={missing_in_predictions[:10]}"
        )

    metadata_columns = [
        "_join_id",
        "plume_id",
        "event_id",
        "availability_signature",
        "label",
        "anchor_sensor",
        "query360_index",
    ]
    joined = predictions.merge(
        manifest[metadata_columns],
        on="_join_id",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", "_manifest"),
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise AssertionError("Strict one-to-one join unexpectedly produced unmatched rows")

    mismatch_examples: dict[str, list[dict[str, Any]]] = {}
    for column in ("plume_id", "availability_signature", "label"):
        left = joined[f"{column}_prediction"].astype(str).str.strip()
        right = joined[f"{column}_manifest"].astype(str).str.strip()
        mismatch = left != right
        if mismatch.any():
            mismatch_examples[column] = (
                joined.loc[
                    mismatch,
                    [
                        "_join_id",
                        f"{column}_prediction",
                        f"{column}_manifest",
                    ],
                ]
                .head(10)
                .to_dict("records")
            )
    if mismatch_examples:
        raise ValueError(
            "Prediction metadata disagrees with canonical manifest: "
            + json.dumps(_jsonable(mismatch_examples), sort_keys=True)
        )

    labels = pd.to_numeric(joined["label_manifest"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(
            f"Labels must be binary; observed {sorted(labels.unique().tolist())}"
        )
    probabilities = pd.to_numeric(joined["probability"], errors="raise")
    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError("Probabilities contain non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must lie in [0, 1]")

    frame = pd.DataFrame(
        {
            "id": joined["_join_id"],
            "query360_index": canonical_id(
                joined["query360_index"], "manifest.query360_index"
            ),
            "plume_id": clean_string(joined["plume_id_manifest"], "plume_id"),
            "event_id": clean_string(joined["event_id"], "event_id"),
            "availability_signature": clean_string(
                joined["availability_signature_manifest"],
                "availability_signature",
            ).map(validate_signature),
            "primary_sensor": clean_string(
                joined["anchor_sensor"], "anchor_sensor"
            ).str.lower(),
            "label": labels.astype(np.int8),
            "probability": probabilities.astype(float),
        }
    )
    unknown_primary = sorted(set(frame["primary_sensor"]).difference(SENSOR_ORDER))
    if unknown_primary:
        raise ValueError(f"Unknown primary/anchor sensors: {unknown_primary}")
    primary_not_available = ~frame.apply(
        lambda row: row["primary_sensor"]
        in row["availability_signature"].split("+"),
        axis=1,
    )
    if primary_not_available.any():
        raise ValueError(
            "Primary sensor absent from availability signature; ids: "
            f"{frame.loc[primary_not_available, 'id'].head(10).tolist()}"
        )

    plume_event_counts = frame.groupby("plume_id", sort=False)["event_id"].nunique()
    if int(plume_event_counts.max()) != 1:
        bad = plume_event_counts[plume_event_counts > 1].head(10).to_dict()
        raise ValueError(f"A plume maps to multiple events: {bad}")

    frame["sensor_count"] = (
        frame["availability_signature"].str.count(re.escape("+")) + 1
    ).astype(np.int8)
    frame["sensor_count_key"] = frame["sensor_count"].map(lambda n: f"n={n}")
    frame["count_primary_key"] = (
        frame["sensor_count_key"] + "|primary=" + frame["primary_sensor"]
    )

    audit = {
        "prediction_rows": len(predictions),
        "manifest_rows": len(manifest),
        "joined_rows": len(frame),
        "join_key": "id",
        "join_cardinality": "one_to_one",
        "prediction_id_unique": True,
        "manifest_id_unique": True,
        "id_sets_identical": True,
        "metadata_fields_verified_equal": [
            "label",
            "plume_id",
            "availability_signature",
        ],
        "canonical_event_source": "manifest.event_id",
        "canonical_primary_sensor_source": "manifest.anchor_sensor",
        "event_count": int(frame["event_id"].nunique()),
        "plume_count": int(frame["plume_id"].nunique()),
        "every_plume_maps_to_exactly_one_event": True,
        "label_counts": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "availability_counts": {
            str(key): int(value)
            for key, value in frame["availability_signature"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return frame, audit


def best_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray, tie_target: float = 0.5
) -> tuple[float, float]:
    """Exact empirical F1 maximizer for the rule probability >= threshold."""
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Threshold fitting requires equally sized non-empty vectors")
    positives = int(y.sum())
    if positives == 0:
        return 1.0, 0.0

    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y)
    cumulative_fp = np.cumsum(1 - sorted_y)
    group_ends = np.flatnonzero(
        np.r_[sorted_p[:-1] != sorted_p[1:], np.array([True])]
    )
    tp = cumulative_tp[group_ends].astype(float)
    fp = cumulative_fp[group_ends].astype(float)
    fn = positives - tp
    denominator = 2.0 * tp + fp + fn
    scores = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator > 0,
    )
    thresholds = sorted_p[group_ends]
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score, atol=1e-12, rtol=0.0))
    distances = np.abs(thresholds[tied] - float(tie_target))
    closest = tied[np.flatnonzero(distances == distances.min())]
    # A higher threshold is the deterministic final tie-break.
    chosen = int(closest[np.argmax(thresholds[closest])])
    return float(thresholds[chosen]), best_score


def build_folds(
    frame: pd.DataFrame, folds: int, seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    event_count = int(frame["event_id"].nunique())
    if event_count < folds:
        raise ValueError(f"Only {event_count} events available for {folds} folds")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    assignments = np.full(len(frame), -1, dtype=np.int16)
    dummy = np.zeros((len(frame), 1), dtype=np.int8)
    audit: list[dict[str, Any]] = []
    for fold, (calibration_indices, held_indices) in enumerate(
        splitter.split(dummy, frame["label"].to_numpy(), frame["event_id"].to_numpy())
    ):
        assignments[held_indices] = fold
        calibration_events = set(frame.iloc[calibration_indices]["event_id"])
        held_events = set(frame.iloc[held_indices]["event_id"])
        calibration_plumes = set(frame.iloc[calibration_indices]["plume_id"])
        held_plumes = set(frame.iloc[held_indices]["plume_id"])
        if calibration_events.intersection(held_events):
            raise AssertionError(f"Event leakage detected in fold {fold}")
        if calibration_plumes.intersection(held_plumes):
            raise AssertionError(f"Plume leakage detected in fold {fold}")
        held_labels = frame.iloc[held_indices]["label"].value_counts()
        if set(held_labels.index) != {0, 1}:
            raise ValueError(f"Held fold {fold} lacks one class: {held_labels.to_dict()}")
        audit.append(
            {
                "fold": fold,
                "calibration_rows": len(calibration_indices),
                "held_rows": len(held_indices),
                "calibration_events": len(calibration_events),
                "held_events": len(held_events),
                "calibration_plumes": len(calibration_plumes),
                "held_plumes": len(held_plumes),
                "held_label_counts": {
                    str(k): int(v) for k, v in held_labels.sort_index().items()
                },
                "event_overlap": 0,
                "plume_overlap": 0,
            }
        )
    if (assignments < 0).any():
        raise AssertionError("Some rows were not assigned exactly one held-out fold")
    return assignments, audit


def fit_group_thresholds(
    calibration: pd.DataFrame,
    group_column: str,
    global_threshold: float,
    min_rows: int,
    min_class_rows: int,
    min_events: int,
    shrink_tau: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    thresholds: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    for group_value, subset in calibration.groupby(group_column, sort=True):
        group_key = str(group_value)
        rows = len(subset)
        positives = int(subset["label"].sum())
        negatives = rows - positives
        events = int(subset["event_id"].nunique())
        eligible = (
            rows >= min_rows
            and positives >= min_class_rows
            and negatives >= min_class_rows
            and events >= min_events
        )
        fallback_reason: str | None = None
        local_threshold: float | None = None
        local_training_f1: float | None = None
        shrink_weight = 0.0
        threshold = global_threshold
        if eligible:
            local_threshold, local_training_f1 = best_f1_threshold(
                subset["label"].to_numpy(),
                subset["probability"].to_numpy(),
                tie_target=global_threshold,
            )
            shrink_weight = rows / (rows + shrink_tau)
            threshold = (
                (1.0 - shrink_weight) * global_threshold
                + shrink_weight * local_threshold
            )
        else:
            failed = []
            if rows < min_rows:
                failed.append(f"rows<{min_rows}")
            if positives < min_class_rows:
                failed.append(f"positives<{min_class_rows}")
            if negatives < min_class_rows:
                failed.append(f"negatives<{min_class_rows}")
            if events < min_events:
                failed.append(f"events<{min_events}")
            fallback_reason = ",".join(failed)

        thresholds[group_key] = float(threshold)
        details[group_key] = {
            "rows": rows,
            "positives": positives,
            "negatives": negatives,
            "events": events,
            "eligible_local_fit": eligible,
            "fallback_reason": fallback_reason,
            "global_threshold": global_threshold,
            "local_threshold_unshrunk": local_threshold,
            "local_training_binary_f1": local_training_f1,
            "shrink_weight": shrink_weight,
            "applied_threshold": threshold,
        }
    return thresholds, details


def make_oof_predictions(
    frame: pd.DataFrame,
    assignments: np.ndarray,
    config: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    strategies = ["global", *STRATEGY_GROUP_COLUMN]
    oof = frame[
        [
            "id",
            "query360_index",
            "event_id",
            "plume_id",
            "availability_signature",
            "sensor_count",
            "primary_sensor",
            "count_primary_key",
            "label",
            "probability",
        ]
    ].copy()
    oof["fold"] = assignments
    for strategy in strategies:
        oof[f"threshold_{strategy}"] = np.nan
        oof[f"prediction_{strategy}"] = -1

    fold_models: list[dict[str, Any]] = []
    for fold in sorted(np.unique(assignments).tolist()):
        calibration = frame.loc[assignments != fold].copy()
        held = frame.loc[assignments == fold].copy()
        held_index = held.index
        global_threshold, global_training_f1 = best_f1_threshold(
            calibration["label"].to_numpy(),
            calibration["probability"].to_numpy(),
            tie_target=0.5,
        )
        oof.loc[held_index, "threshold_global"] = global_threshold
        oof.loc[held_index, "prediction_global"] = (
            held["probability"].to_numpy() >= global_threshold
        ).astype(np.int8)

        fold_record: dict[str, Any] = {
            "fold": fold,
            "calibration_rows": len(calibration),
            "held_rows": len(held),
            "global": {
                "threshold": global_threshold,
                "calibration_binary_f1": global_training_f1,
            },
            "conditioned": {},
        }
        for strategy, group_column in STRATEGY_GROUP_COLUMN.items():
            thresholds, details = fit_group_thresholds(
                calibration=calibration,
                group_column=group_column,
                global_threshold=global_threshold,
                min_rows=config.min_group_rows,
                min_class_rows=config.min_group_class_rows,
                min_events=config.min_group_events,
                shrink_tau=config.shrink_tau,
            )
            held_keys = held[group_column].astype(str)
            held_thresholds = held_keys.map(thresholds).fillna(global_threshold)
            oof.loc[held_index, f"threshold_{strategy}"] = (
                held_thresholds.to_numpy()
            )
            oof.loc[held_index, f"prediction_{strategy}"] = (
                held["probability"].to_numpy() >= held_thresholds.to_numpy()
            ).astype(np.int8)
            unseen = sorted(set(held_keys).difference(thresholds))
            fold_record["conditioned"][strategy] = {
                "group_column": group_column,
                "unseen_held_groups_falling_back_to_global": unseen,
                "groups": details,
            }
        fold_models.append(fold_record)

    threshold_columns = [f"threshold_{strategy}" for strategy in strategies]
    prediction_columns = [f"prediction_{strategy}" for strategy in strategies]
    if not np.isfinite(oof[threshold_columns].to_numpy()).all():
        raise AssertionError("OOF thresholds contain missing/non-finite values")
    if not oof[prediction_columns].isin([0, 1]).all().all():
        raise AssertionError("OOF predictions are not complete binary values")
    return oof, fold_models


def metric_pair(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "binary_f1": float(
            f1_score(labels, predictions, average="binary", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
    }


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def clustered_bootstrap(
    oof: pd.DataFrame,
    strategies: list[str],
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    group_indices = [
        indices.to_numpy(dtype=np.int64)
        for _, indices in oof.groupby("event_id", sort=True).groups.items()
    ]
    rng = np.random.default_rng(seed)
    distributions: dict[str, dict[str, list[float]]] = {
        strategy: {"binary_f1": [], "macro_f1": []} for strategy in strategies
    }
    valid = 0
    attempts = 0
    maximum_attempts = max(replicates * 10, replicates + 100)
    labels_all = oof["label"].to_numpy(dtype=np.int8)
    predictions_all = {
        strategy: oof[f"prediction_{strategy}"].to_numpy(dtype=np.int8)
        for strategy in strategies
    }
    while valid < replicates and attempts < maximum_attempts:
        attempts += 1
        sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        sampled_indices = np.concatenate([group_indices[i] for i in sampled_groups])
        sampled_labels = labels_all[sampled_indices]
        if np.unique(sampled_labels).size != 2:
            continue
        for strategy in strategies:
            values = metric_pair(
                sampled_labels, predictions_all[strategy][sampled_indices]
            )
            for metric, value in values.items():
                distributions[strategy][metric].append(value)
        valid += 1
    if valid != replicates:
        raise RuntimeError(
            f"Only obtained {valid}/{replicates} valid two-class bootstrap replicates"
        )

    metrics: dict[str, Any] = {}
    point_values = {
        strategy: metric_pair(labels_all, predictions_all[strategy])
        for strategy in strategies
    }
    for strategy in strategies:
        metrics[strategy] = {}
        for metric in ("binary_f1", "macro_f1"):
            low, high = percentile_interval(distributions[strategy][metric])
            metrics[strategy][metric] = {
                "point": point_values[strategy][metric],
                "ci_95_low": low,
                "ci_95_high": high,
                "bootstrap_method": (
                    "canonical-event cluster percentile, OOF predictions fixed"
                ),
            }

    deltas: dict[str, Any] = {}
    for strategy in strategies:
        if strategy == "global":
            continue
        deltas[strategy] = {}
        for metric in ("binary_f1", "macro_f1"):
            values = np.asarray(distributions[strategy][metric]) - np.asarray(
                distributions["global"][metric]
            )
            low, high = percentile_interval(values)
            deltas[strategy][metric] = {
                "point": (
                    point_values[strategy][metric]
                    - point_values["global"][metric]
                ),
                "ci_95_low": low,
                "ci_95_high": high,
                "paired_bootstrap_method": (
                    "same resampled canonical events for both OOF strategies"
                ),
            }
    return metrics, deltas, attempts


def subgroup_oof_metrics(oof: pd.DataFrame, strategies: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for signature, subset in oof.groupby("availability_signature", sort=True):
        labels = subset["label"].to_numpy()
        record: dict[str, Any] = {
            "rows": len(subset),
            "events": int(subset["event_id"].nunique()),
            "positives": int(labels.sum()),
            "negatives": int(len(labels) - labels.sum()),
            "strategies": {},
        }
        for strategy in strategies:
            record["strategies"][strategy] = metric_pair(
                labels, subset[f"prediction_{strategy}"].to_numpy()
            )
        output[str(signature)] = record
    return output


def render_markdown(result: dict[str, Any]) -> str:
    metrics = result["oof_metrics_with_event_cluster_ci"]
    deltas = result["paired_deltas_vs_global"]
    rows = []
    for strategy in result["strategy_order"]:
        binary = metrics[strategy]["binary_f1"]
        macro = metrics[strategy]["macro_f1"]
        if strategy == "global":
            delta_binary = "reference"
            delta_macro = "reference"
        else:
            db = deltas[strategy]["binary_f1"]
            dm = deltas[strategy]["macro_f1"]
            delta_binary = (
                f"{db['point']:+.4f} [{db['ci_95_low']:+.4f}, "
                f"{db['ci_95_high']:+.4f}]"
            )
            delta_macro = (
                f"{dm['point']:+.4f} [{dm['ci_95_low']:+.4f}, "
                f"{dm['ci_95_high']:+.4f}]"
            )
        rows.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    (
                        f"{binary['point']:.4f} "
                        f"[{binary['ci_95_low']:.4f}, {binary['ci_95_high']:.4f}]"
                    ),
                    (
                        f"{macro['point']:.4f} "
                        f"[{macro['ci_95_low']:.4f}, {macro['ci_95_high']:.4f}]"
                    ),
                    delta_binary,
                    delta_macro,
                ]
            )
            + " |"
        )

    best_binary = max(
        result["strategy_order"],
        key=lambda name: metrics[name]["binary_f1"]["point"],
    )
    best_macro = max(
        result["strategy_order"],
        key=lambda name: metrics[name]["macro_f1"]["point"],
    )
    audit = result["join_audit"]
    protocol = result["protocol"]
    fold_lines = []
    for fold in result["fold_audit"]:
        fold_lines.append(
            f"| {fold['fold']} | {fold['calibration_rows']} | "
            f"{fold['held_rows']} | {fold['held_events']} | "
            f"{fold['held_plumes']} | "
            f"{fold['held_label_counts'].get('0', 0)}/"
            f"{fold['held_label_counts'].get('1', 0)} |"
        )

    return f"""# Availability-conditioned calibration: strict dev-only OOF audit

## Scope and interpretation

This is a **post-hoc calibration study on one frozen model's dev probabilities**.
It is not a dual-axis architecture gain, not a representation/pretraining gain,
and not a final test-set result. No model weights or logits were changed.
OOF splitting removes held-row leakage from *threshold fitting only*; it cannot
undo any optimism if the checkpoint, epoch, or model hyperparameters that produced
`dev_predictions_best.csv` were themselves selected using this same dev set.

Every row was held out while its decision threshold was fitted on the other four
folds. Folds are grouped by canonical `event_id`, which also makes plumes disjoint
because the manifest audit established that every plume maps to exactly one event.
No path containing a `test` or `sealed` token was accepted or read.

## OOF results

95% intervals use {protocol['bootstrap_replicates']} canonical-event cluster
bootstrap replicates with the already-generated OOF predictions held fixed.
Paired deltas use identical sampled events for each strategy and the global
calibrator.

| Strategy | Binary F1 (95% CI) | Macro F1 (95% CI) | Δ binary vs global (95% CI) | Δ macro vs global (95% CI) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Highest observed dev-only OOF binary F1: `{best_binary}`. Highest observed
dev-only OOF macro F1: `{best_macro}`. These labels are descriptive, not an
independent model-selection claim.

## Calibrators

- `global`: one empirical binary-F1 threshold fitted on the other four folds.
- `availability_signature`: a local threshold for each exact sensor signature.
- `sensor_count`: a coarse local threshold based only on number of available sensors.
- `primary_sensor`: a coarse local threshold based on canonical `anchor_sensor`.
- `count_primary`: a coarse local threshold for `(sensor count, anchor sensor)`.

For every conditioned calibrator, a group must have at least
{protocol['min_group_rows']} rows, {protocol['min_group_class_rows']} rows of each
class, and {protocol['min_group_events']} events in the four calibration folds.
Otherwise it falls back to that fold's global threshold. Eligible local
thresholds are shrunk toward global with
`w = n / (n + {protocol['shrink_tau']})`. These rules were fixed before reading
the held fold.

## Identity and leakage audit

- Frozen prediction rows: {audit['prediction_rows']:,}
- Canonical manifest rows: {audit['manifest_rows']:,}
- Exact one-to-one `id` join: {audit['joined_rows']:,}; ID sets identical
- Canonical events/plumes: {audit['event_count']:,} / {audit['plume_count']:,}
- Prediction metadata checked against manifest: label, plume ID, availability signature
- Event and plume overlap between calibration and held portions: zero in every fold
- Input SHA-256 hashes and all per-fold fitted thresholds are retained in the JSON
- Test/sealed inputs read: **none**

| Held fold | Calibration rows | Held rows | Held events | Held plumes | Held labels 0/1 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_lines)}

## Correct research claim

This audit can support only the statement that availability metadata may (or may
not) improve **decision calibration** for a fixed classifier on development data.
It cannot support a claim that two-axis attention or invisible-band pretraining
improved representation quality. Any threshold policy selected from this study
must be frozen and evaluated once on a separately authorized final set before
making a generalization claim.
"""


def main() -> None:
    args = parse_args()
    if args.min_group_rows < 1:
        raise ValueError("--min-group-rows must be positive")
    if args.min_group_class_rows < 1:
        raise ValueError("--min-group-class-rows must be positive")
    if args.min_group_events < 1:
        raise ValueError("--min-group-events must be positive")
    if not math.isfinite(args.shrink_tau) or args.shrink_tau < 0:
        raise ValueError("--shrink-tau must be finite and non-negative")

    prediction_path = assert_dev_only_input(args.predictions, "predictions")
    manifest_path = assert_dev_only_input(args.manifest, "manifest")
    frame, join_audit = read_and_join(prediction_path, manifest_path)
    assignments, fold_audit = build_folds(frame, args.folds, args.seed)
    oof, fold_models = make_oof_predictions(frame, assignments, args)
    strategy_order = ["global", *STRATEGY_GROUP_COLUMN]
    metrics, deltas, bootstrap_attempts = clustered_bootstrap(
        oof=oof,
        strategies=strategy_order,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "post_hoc_decision_threshold_calibration",
        "interpretation_guardrail": (
            "Dev-only OOF calibration of frozen probabilities; not a dual-axis, "
            "training, representation, or pretraining gain; no final-test claim."
        ),
        "selection_bias_guardrail": (
            "OOF controls threshold-fitting leakage only. It does not remove bias "
            "from any checkpoint/epoch/hyperparameter selection previously performed "
            "with this same dev set."
        ),
        "inputs": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": file_sha256(prediction_path),
            },
            "canonical_dev_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
            },
            "test_or_sealed_input_read": False,
        },
        "join_audit": join_audit,
        "protocol": {
            "folds": args.folds,
            "fold_assignment": (
                "StratifiedGroupKFold(shuffle=True), canonical event_id groups"
            ),
            "seed": args.seed,
            "held_threshold_rule": (
                "Every held fold uses thresholds fitted only on the other folds"
            ),
            "positive_class": 1,
            "global_objective": "empirical binary F1 on calibration folds",
            "local_objective": "empirical binary F1 within calibration subgroup",
            "threshold_prediction_rule": "probability >= threshold",
            "threshold_tie_break": (
                "closest to parent/global target, then higher threshold"
            ),
            "min_group_rows": args.min_group_rows,
            "min_group_class_rows": args.min_group_class_rows,
            "min_group_events": args.min_group_events,
            "shrink_tau": args.shrink_tau,
            "shrink_formula": (
                "w=n/(n+tau); applied=(1-w)*global+w*local; ineligible/unseen=>global"
            ),
            "bootstrap_unit": "canonical event_id",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.seed + 1,
            "bootstrap_attempts": bootstrap_attempts,
            "bootstrap_refits_calibrators": False,
        },
        "strategy_order": strategy_order,
        "strategy_group_columns": STRATEGY_GROUP_COLUMN,
        "fold_audit": fold_audit,
        "fold_calibrators": fold_models,
        "oof_metrics_with_event_cluster_ci": metrics,
        "paired_deltas_vs_global": deltas,
        "oof_metrics_by_availability_signature": subgroup_oof_metrics(
            oof, strategy_order
        ),
    }
    result = _jsonable(result)

    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.output_predictions.with_suffix(
        args.output_predictions.suffix + ".tmp"
    )
    oof.to_csv(temporary_csv, index=False)
    temporary_csv.replace(args.output_predictions)
    atomic_write_text(
        args.output_json,
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(args.output_md, render_markdown(result))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_md": str(args.output_md.resolve()),
                "output_predictions": str(args.output_predictions.resolve()),
                "rows": len(oof),
                "events": int(oof["event_id"].nunique()),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
