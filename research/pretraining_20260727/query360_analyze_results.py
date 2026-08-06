#!/usr/bin/env python3
"""Paired, canonical-event bootstrap analysis for Query360 experiments.

Only completed experiment summaries and their
``validation_best_ap_predictions.csv`` files are opened.  Train, validation,
test, sealed, and holdout manifests are never read by this script.

The bootstrap resamples canonical acquisition/event IDs with replacement and
brings every paired row belonging to a sampled event along with it.  The same
event draws are shared by every model and contrast, including the factorial
difference-in-differences contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "query360-canonical-event-bootstrap-v1"
CANONICAL_EVENT_RULE = "strip-terminal-hyphen-alphanumeric-suffix-v1"
EVENT_SUFFIX_RE = re.compile(r"-[A-Za-z0-9]+$")
BANNED_PATH_SUBSTRINGS = ("test", "sealed", "holdout")
METRIC_NAMES = ("ap", "auc", "macro_f1_at_0_5")
MIN_BOOTSTRAP_REPEATS = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_727
EXPECTED_FROZEN_CONDITIONS = {
    "panopticon_pretrained",
    "random_frozen",
}
EXPECTED_ONLINE_CONDITIONS = {
    "panopticon_pretrained",
    "scratch",
}
REQUIRED_PREDICTION_COLUMNS = (
    "id",
    "plume_id",
    "cluster_id",
    "macro_region_id",
    "availability_signature",
    "label",
    "probability",
    "prediction_at_0_5",
    "arm",
    "epoch",
)
PAIR_METADATA_COLUMNS = (
    "plume_id",
    "event_id",
    "cluster_id",
    "macro_region_id",
    "availability_signature",
    "label",
)
PRIMARY_ARM_CONTRASTS = (
    ("transient_query", "current_only"),
    ("scale_aware_transient_query", "current_only"),
    ("transient_query", "history_shuffle_train"),
    ("scale_aware_transient_query", "transient_query"),
)


class AnalysisError(RuntimeError):
    """Raised when artifact provenance or row pairing is not trustworthy."""


@dataclass(frozen=True)
class PredictionArtifact:
    family: str
    condition: str
    seed: int
    arm: str
    epoch: int
    summary_path: Path
    prediction_path: Path
    summary_sha256: str
    prediction_sha256: str
    frame: pd.DataFrame
    point_metrics: Mapping[str, float]

    @property
    def artifact_id(self) -> str:
        return f"{self.family}|{self.condition}|seed={self.seed}|{self.arm}"


@dataclass(frozen=True)
class SummaryBundle:
    family: str
    condition: str
    path: Path
    sha256: str
    validation_manifest_sha256: str
    validation_rows: int
    dataset_contract: Mapping[str, Any]
    training_contract: Mapping[str, Any]
    artifacts: tuple[PredictionArtifact, ...]


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    family: str
    seed: int
    kind: str
    terms: tuple[tuple[str, float], ...]
    left_artifact: Optional[str] = None
    right_artifact: Optional[str] = None
    aggregate_method: Optional[str] = None
    aggregate_seed_count: Optional[int] = None

    @property
    def formula(self) -> str:
        rendered: list[str] = []
        for artifact_id, coefficient in self.terms:
            sign = "+" if coefficient >= 0 else "-"
            magnitude = abs(float(coefficient))
            term = artifact_id if magnitude == 1.0 else f"{magnitude:g}*{artifact_id}"
            if not rendered:
                rendered.append(term if coefficient >= 0 else f"-{term}")
            else:
                rendered.append(f" {sign} {term}")
        return "".join(rendered)


def assert_safe_path(path: os.PathLike[str] | str, *, purpose: str) -> Path:
    text = os.fspath(path).strip()
    if not text:
        raise AnalysisError(f"Empty {purpose} path is not accepted.")
    lexical = Path(text).expanduser()
    absolute = Path(os.path.abspath(os.fspath(lexical)))
    for candidate in (lexical, absolute):
        for component in candidate.parts:
            folded = component.casefold()
            forbidden = next(
                (token for token in BANNED_PATH_SUBSTRINGS if token in folded),
                None,
            )
            if forbidden is not None:
                raise AnalysisError(
                    f"Refusing {purpose} containing forbidden substring "
                    f"{forbidden!r}: {candidate}"
                )
    return absolute


def sha256_file(path: os.PathLike[str] | str) -> str:
    safe = assert_safe_path(path, purpose="input artifact")
    digest = hashlib.sha256()
    with safe.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, value: Any) -> None:
    path = assert_safe_path(path, purpose="JSON output")
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


def _atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path = assert_safe_path(path, purpose="CSV output")
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


def canonical_event_id(plume_id: str) -> str:
    value = EVENT_SUFFIX_RE.sub("", str(plume_id).strip())
    if not value:
        raise AnalysisError(
            f"Canonical event derivation produced an empty ID from {plume_id!r}."
        )
    return value


def _binary_metrics_arrays(
    labels: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Return AP, AUROC, and fixed-0.5 macro-F1 using grouped score ties."""

    target = np.asarray(labels, dtype=np.int8).reshape(-1)
    score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if target.shape != score.shape or target.size == 0:
        raise AnalysisError("Metric inputs must be non-empty matching vectors.")
    if not np.isfinite(score).all():
        raise AnalysisError("Metric probabilities contain non-finite values.")
    unique = np.unique(target)
    if not np.array_equal(unique, np.asarray([0, 1], dtype=np.int8)):
        raise AnalysisError(
            "AP/AUROC bootstrap samples must contain both binary classes."
        )

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order]
    cumulative_tp = np.cumsum(sorted_target == 1, dtype=np.int64)
    cumulative_fp = np.cumsum(sorted_target == 0, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_score[1:] != sorted_score[:-1], True]
    )
    tp_curve = cumulative_tp[group_ends].astype(np.float64)
    fp_curve = cumulative_fp[group_ends].astype(np.float64)
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))

    recall = tp_curve / positives
    precision = tp_curve / np.maximum(tp_curve + fp_curve, 1.0)
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    tpr = np.r_[0.0, tp_curve / positives]
    fpr = np.r_[0.0, fp_curve / negatives]
    auc = float(np.trapz(tpr, fpr))

    prediction = score >= 0.5
    positive = target == 1
    tp = float(np.sum(prediction & positive))
    fp = float(np.sum(prediction & ~positive))
    fn = float(np.sum(~prediction & positive))
    tn = float(np.sum(~prediction & ~positive))
    f1_positive = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1_negative = (2.0 * tn / (2.0 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    macro_f1 = float((f1_positive + f1_negative) / 2.0)
    result = np.asarray([ap, auc, macro_f1], dtype=np.float64)
    if not np.isfinite(result).all():
        raise AnalysisError("Metric computation produced a non-finite value.")
    return result


def binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    values = _binary_metrics_arrays(
        np.asarray(labels), np.asarray(probabilities)
    )
    return {
        metric: float(values[index])
        for index, metric in enumerate(METRIC_NAMES)
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    safe = assert_safe_path(path, purpose="summary")
    if not safe.is_file():
        raise FileNotFoundError(safe)
    with safe.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise AnalysisError(f"Summary is not a JSON object: {safe}")
    return payload


def _clean_string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(str).str.strip()
    if values.eq("").any():
        bad = np.flatnonzero(values.eq("").to_numpy())[:10].tolist()
        raise AnalysisError(f"Prediction column {column!r} is blank at rows {bad}.")
    return values


def _load_prediction_csv(
    path: Path,
    *,
    expected_arm: str,
    expected_condition: str,
    expected_seed: Optional[int],
    expected_epoch: int,
) -> pd.DataFrame:
    safe = assert_safe_path(path, purpose="validation prediction CSV")
    if not safe.is_file():
        raise FileNotFoundError(safe)
    frame = pd.read_csv(
        safe,
        dtype={
            "id": str,
            "plume_id": str,
            "cluster_id": str,
            "macro_region_id": str,
            "availability_signature": str,
            "arm": str,
            "condition": str,
        },
        keep_default_na=False,
        low_memory=False,
    )
    missing = [name for name in REQUIRED_PREDICTION_COLUMNS if name not in frame]
    if missing:
        raise AnalysisError(f"{safe}: missing prediction columns {missing}.")
    if frame.empty:
        raise AnalysisError(f"{safe}: prediction CSV is empty.")

    for column in (
        "id",
        "plume_id",
        "cluster_id",
        "macro_region_id",
        "availability_signature",
        "arm",
    ):
        frame[column] = _clean_string_column(frame, column)
    if frame["id"].duplicated().any():
        examples = frame.loc[frame["id"].duplicated(False), "id"].head(10).tolist()
        raise AnalysisError(f"{safe}: duplicate row IDs are not pairable: {examples}.")

    label_numeric = pd.to_numeric(frame["label"], errors="coerce")
    if label_numeric.isna().any() or not label_numeric.isin([0, 1]).all():
        raise AnalysisError(f"{safe}: labels must be exact binary integers.")
    frame["label"] = label_numeric.astype(np.int8)
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if (
        probability.isna().any()
        or not np.isfinite(probability.to_numpy(dtype=np.float64)).all()
        or ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise AnalysisError(f"{safe}: probabilities must be finite and in [0,1].")
    frame["probability"] = probability.astype(np.float64)

    predicted = pd.to_numeric(frame["prediction_at_0_5"], errors="coerce")
    expected_predicted = (frame["probability"] >= 0.5).astype(np.int8)
    if predicted.isna().any() or not np.array_equal(
        predicted.to_numpy(dtype=np.int64),
        expected_predicted.to_numpy(dtype=np.int64),
    ):
        raise AnalysisError(f"{safe}: prediction_at_0_5 disagrees with probability.")
    if set(frame["arm"]) != {expected_arm}:
        raise AnalysisError(
            f"{safe}: arm column does not equal expected {expected_arm!r}."
        )

    if "condition" in frame:
        frame["condition"] = _clean_string_column(frame, "condition")
        if set(frame["condition"]) != {expected_condition}:
            raise AnalysisError(
                f"{safe}: condition column does not equal {expected_condition!r}."
            )
    if expected_seed is not None:
        if "seed" not in frame:
            raise AnalysisError(f"{safe}: frozen prediction lacks seed column.")
        seeds = pd.to_numeric(frame["seed"], errors="coerce")
        if seeds.isna().any() or set(seeds.astype(int)) != {int(expected_seed)}:
            raise AnalysisError(f"{safe}: seed column does not match {expected_seed}.")
    epochs = pd.to_numeric(frame["epoch"], errors="coerce")
    if epochs.isna().any() or set(epochs.astype(int)) != {int(expected_epoch)}:
        raise AnalysisError(f"{safe}: epoch column does not match {expected_epoch}.")

    derived_event = frame["plume_id"].map(canonical_event_id)
    if "event_id" in frame:
        declared_event = _clean_string_column(frame, "event_id")
        if not declared_event.equals(derived_event):
            mismatch = np.flatnonzero(
                declared_event.to_numpy() != derived_event.to_numpy()
            )[:10].tolist()
            raise AnalysisError(
                f"{safe}: declared event_id violates {CANONICAL_EVENT_RULE} "
                f"at rows {mismatch}."
            )
    frame["event_id"] = derived_event
    return frame.sort_values("id", kind="mergesort").reset_index(drop=True)


def validate_paired_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Fail unless two normalized prediction tables have identical paired rows."""

    if len(reference) != len(candidate):
        raise AnalysisError(
            f"{context}: paired row counts differ: {len(reference)} vs "
            f"{len(candidate)}."
        )
    reference_ids = reference["id"].astype(str).tolist()
    candidate_ids = candidate["id"].astype(str).tolist()
    if reference_ids != candidate_ids:
        left_only = sorted(set(reference_ids) - set(candidate_ids))[:10]
        right_only = sorted(set(candidate_ids) - set(reference_ids))[:10]
        raise AnalysisError(
            f"{context}: paired row IDs differ; reference_only={left_only}, "
            f"candidate_only={right_only}."
        )
    for column in PAIR_METADATA_COLUMNS:
        left = reference[column].to_numpy()
        right = candidate[column].to_numpy()
        if not np.array_equal(left, right):
            mismatch = np.flatnonzero(left != right)[:10].tolist()
            raise AnalysisError(
                f"{context}: paired {column} values differ at rows {mismatch}."
            )


def _validate_summary_metrics(
    *,
    prediction_path: Path,
    frame: pd.DataFrame,
    validation: Mapping[str, Any],
    tolerance: float = 1e-7,
) -> dict[str, float]:
    overall = validation.get("overall")
    if not isinstance(overall, Mapping):
        raise AnalysisError(f"{prediction_path}: summary lacks validation.overall.")
    if int(overall.get("rows", -1)) != len(frame):
        raise AnalysisError(
            f"{prediction_path}: summary/prediction validation rows differ."
        )
    computed = binary_metrics(frame["label"], frame["probability"])
    for metric, value in computed.items():
        expected = overall.get(metric)
        if expected is None or not np.isfinite(float(expected)):
            raise AnalysisError(
                f"{prediction_path}: summary metric {metric} is missing/non-finite."
            )
        if not np.isclose(value, float(expected), atol=tolerance, rtol=tolerance):
            raise AnalysisError(
                f"{prediction_path}: recomputed {metric}={value} differs from "
                f"summary={expected}."
            )
    return computed


def _require_no_external_test_read(summary: Mapping[str, Any], *, path: Path) -> None:
    safety = summary.get("safety")
    if not isinstance(safety, Mapping):
        raise AnalysisError(f"{path}: summary lacks a safety contract.")
    if safety.get("external_test_manifest_read") is not False:
        raise AnalysisError(
            f"{path}: external_test_manifest_read must be explicitly false."
        )


def load_frozen_summary(path: os.PathLike[str] | str) -> SummaryBundle:
    summary_path = assert_safe_path(path, purpose="frozen summary")
    summary = _read_json(summary_path)
    if summary.get("schema_version") != "query360-head-summary-v1":
        raise AnalysisError(f"{summary_path}: unsupported frozen summary schema.")
    _require_no_external_test_read(summary, path=summary_path)
    condition = str(summary.get("encoder", {}).get("condition", "")).strip()
    if condition not in EXPECTED_FROZEN_CONDITIONS:
        raise AnalysisError(
            f"{summary_path}: unexpected frozen encoder condition {condition!r}."
        )
    dataset_contract = summary.get("dataset_contract")
    training_contract = summary.get("training_contract")
    if not isinstance(dataset_contract, Mapping) or not isinstance(
        training_contract, Mapping
    ):
        raise AnalysisError(f"{summary_path}: missing formal dataset/training contract.")
    inner_val_contract = dataset_contract.get("inner_val")
    if not isinstance(inner_val_contract, Mapping):
        raise AnalysisError(f"{summary_path}: missing inner_val dataset contract.")
    validation_sha = str(inner_val_contract.get("manifest_sha256", "")).strip()
    validation_rows = int(inner_val_contract.get("rows", -1))
    if len(validation_sha) != 64 or validation_rows <= 0:
        raise AnalysisError(f"{summary_path}: invalid inner_val provenance contract.")

    arms = tuple(str(value) for value in summary.get("arms", ()))
    seeds = tuple(int(value) for value in summary.get("seeds", ()))
    if not arms or len(set(arms)) != len(arms):
        raise AnalysisError(f"{summary_path}: frozen arms are missing/duplicated.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise AnalysisError(f"{summary_path}: frozen seeds are missing/duplicated.")
    if list(arms) != list(training_contract.get("arms", ())):
        raise AnalysisError(f"{summary_path}: summary/training-contract arms differ.")
    if list(seeds) != list(training_contract.get("seeds", ())):
        raise AnalysisError(f"{summary_path}: summary/training-contract seeds differ.")
    seed_results_raw = summary.get("seed_results")
    if not isinstance(seed_results_raw, Sequence):
        raise AnalysisError(f"{summary_path}: seed_results is malformed.")
    seed_results = {
        int(result["seed"]): result
        for result in seed_results_raw
        if isinstance(result, Mapping) and "seed" in result
    }
    if set(seed_results) != set(seeds):
        raise AnalysisError(f"{summary_path}: seed_results do not match seeds.")

    summary_digest = sha256_file(summary_path)
    artifacts: list[PredictionArtifact] = []
    for seed in seeds:
        result = seed_results[seed]
        result_arms = result.get("arms")
        if not isinstance(result_arms, Mapping) or set(result_arms) != set(arms):
            raise AnalysisError(
                f"{summary_path}: seed {seed} arm results are incomplete."
            )
        for arm in arms:
            best = result_arms[arm]
            if not isinstance(best, Mapping):
                raise AnalysisError(f"{summary_path}: malformed best result for {arm}.")
            epoch = int(best.get("epoch", -1))
            if epoch <= 0:
                raise AnalysisError(f"{summary_path}: invalid best epoch for {arm}.")
            prediction_path = (
                summary_path.parent
                / f"seed_{seed}"
                / arm
                / "validation_best_ap_predictions.csv"
            )
            frame = _load_prediction_csv(
                prediction_path,
                expected_arm=arm,
                expected_condition=condition,
                expected_seed=seed,
                expected_epoch=epoch,
            )
            if len(frame) != validation_rows:
                raise AnalysisError(
                    f"{prediction_path}: rows differ from frozen dataset contract."
                )
            metrics = _validate_summary_metrics(
                prediction_path=prediction_path,
                frame=frame,
                validation=best.get("validation", {}),
            )
            artifacts.append(
                PredictionArtifact(
                    family="frozen",
                    condition=condition,
                    seed=seed,
                    arm=arm,
                    epoch=epoch,
                    summary_path=summary_path,
                    prediction_path=prediction_path,
                    summary_sha256=summary_digest,
                    prediction_sha256=sha256_file(prediction_path),
                    frame=frame,
                    point_metrics=metrics,
                )
            )
    return SummaryBundle(
        family="frozen",
        condition=condition,
        path=summary_path,
        sha256=summary_digest,
        validation_manifest_sha256=validation_sha,
        validation_rows=validation_rows,
        dataset_contract=dict(dataset_contract),
        training_contract=dict(training_contract),
        artifacts=tuple(artifacts),
    )


def load_online_summary(path: os.PathLike[str] | str) -> SummaryBundle:
    summary_path = assert_safe_path(path, purpose="online summary")
    summary = _read_json(summary_path)
    if summary.get("schema_version") != "query360-online-summary-v1":
        raise AnalysisError(f"{summary_path}: unsupported online summary schema.")
    _require_no_external_test_read(summary, path=summary_path)
    if summary.get("full_backbone_train") is not True:
        raise AnalysisError(f"{summary_path}: online summary is not full-backbone train.")
    condition = str(summary.get("condition", "")).strip()
    if condition not in EXPECTED_ONLINE_CONDITIONS:
        raise AnalysisError(
            f"{summary_path}: unexpected online condition {condition!r}."
        )
    training_contract = summary.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise AnalysisError(f"{summary_path}: missing online training contract.")
    if int(training_contract.get("max_eval_steps", -1)) != 0:
        raise AnalysisError(
            f"{summary_path}: formal analysis requires a full inner-val evaluation."
        )
    seed = int(training_contract.get("batch_seed", -1))
    if seed < 0:
        raise AnalysisError(f"{summary_path}: invalid online batch seed.")
    val_manifest = summary.get("inner_val_manifest")
    train_manifest = summary.get("train_manifest")
    if not isinstance(val_manifest, Mapping) or not isinstance(train_manifest, Mapping):
        raise AnalysisError(f"{summary_path}: missing online manifest provenance.")
    # Validate lexical safety but deliberately do not open either manifest.
    assert_safe_path(
        str(val_manifest.get("path", "")),
        purpose="online inner-val manifest metadata",
    )
    assert_safe_path(
        str(train_manifest.get("path", "")),
        purpose="online train manifest metadata",
    )
    validation_sha = str(val_manifest.get("sha256", "")).strip()
    validation_rows = int(val_manifest.get("rows", -1))
    if len(validation_sha) != 64 or validation_rows <= 0:
        raise AnalysisError(f"{summary_path}: invalid online inner-val provenance.")

    arm_results = summary.get("arms")
    if not isinstance(arm_results, Mapping) or not arm_results:
        raise AnalysisError(f"{summary_path}: online arm results are malformed.")
    requested_arms = tuple(
        str(value) for value in training_contract.get("requested_arms", ())
    )
    if set(arm_results) != set(requested_arms):
        raise AnalysisError(f"{summary_path}: requested/result online arms differ.")

    summary_digest = sha256_file(summary_path)
    artifacts: list[PredictionArtifact] = []
    for arm in requested_arms:
        best = arm_results[arm]
        if not isinstance(best, Mapping):
            raise AnalysisError(f"{summary_path}: malformed online best result.")
        epoch = int(best.get("epoch", -1))
        if epoch <= 0:
            raise AnalysisError(f"{summary_path}: invalid best epoch for {arm}.")
        prediction_path = (
            summary_path.parent / arm / "validation_best_ap_predictions.csv"
        )
        frame = _load_prediction_csv(
            prediction_path,
            expected_arm=arm,
            expected_condition=condition,
            expected_seed=None,
            expected_epoch=epoch,
        )
        if len(frame) != validation_rows:
            raise AnalysisError(
                f"{prediction_path}: rows differ from online manifest contract."
            )
        metrics = _validate_summary_metrics(
            prediction_path=prediction_path,
            frame=frame,
            validation=best.get("validation", {}),
        )
        artifacts.append(
            PredictionArtifact(
                family="online",
                condition=condition,
                seed=seed,
                arm=arm,
                epoch=epoch,
                summary_path=summary_path,
                prediction_path=prediction_path,
                summary_sha256=summary_digest,
                prediction_sha256=sha256_file(prediction_path),
                frame=frame,
                point_metrics=metrics,
            )
        )
    dataset_contract = {
        "train_manifest": dict(train_manifest),
        "inner_val_manifest": dict(val_manifest),
    }
    return SummaryBundle(
        family="online",
        condition=condition,
        path=summary_path,
        sha256=summary_digest,
        validation_manifest_sha256=validation_sha,
        validation_rows=validation_rows,
        dataset_contract=dataset_contract,
        training_contract=dict(training_contract),
        artifacts=tuple(artifacts),
    )


def _validate_bundles(
    frozen: Sequence[SummaryBundle],
    online: Sequence[SummaryBundle],
) -> tuple[PredictionArtifact, ...]:
    if {bundle.condition for bundle in frozen} != EXPECTED_FROZEN_CONDITIONS:
        raise AnalysisError(
            "Frozen inputs must contain exactly panopticon_pretrained and "
            "random_frozen summaries."
        )
    if {bundle.condition for bundle in online} != EXPECTED_ONLINE_CONDITIONS:
        raise AnalysisError(
            "Online inputs must contain exactly panopticon_pretrained and "
            "scratch summaries."
        )
    if len(frozen) != 2 or len(online) != 2:
        raise AnalysisError("Formal analysis requires exactly two summaries per family.")

    if frozen[0].dataset_contract != frozen[1].dataset_contract:
        raise AnalysisError("Frozen summaries do not have identical dataset contracts.")
    if frozen[0].training_contract != frozen[1].training_contract:
        raise AnalysisError("Frozen summaries do not have identical training contracts.")
    if (
        online[0].dataset_contract["train_manifest"]
        != online[1].dataset_contract["train_manifest"]
        or online[0].dataset_contract["inner_val_manifest"]
        != online[1].dataset_contract["inner_val_manifest"]
    ):
        raise AnalysisError("Online summaries do not use identical manifests.")
    # Different backbone learning rates are condition-specific by design; every
    # other online training setting must match.
    online_contracts = []
    for bundle in online:
        contract = dict(bundle.training_contract)
        contract.pop("backbone_lr", None)
        online_contracts.append(contract)
    if online_contracts[0] != online_contracts[1]:
        raise AnalysisError(
            "Online summaries differ in settings other than backbone learning rate."
        )

    all_bundles = tuple(frozen) + tuple(online)
    validation_shas = {
        bundle.validation_manifest_sha256 for bundle in all_bundles
    }
    validation_rows = {bundle.validation_rows for bundle in all_bundles}
    if len(validation_shas) != 1 or len(validation_rows) != 1:
        raise AnalysisError(
            "Frozen/online summaries do not share one inner-val manifest and row count."
        )
    artifacts = tuple(
        artifact for bundle in all_bundles for artifact in bundle.artifacts
    )
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise AnalysisError(f"Duplicate artifact identities: {artifact_ids}.")
    if not artifacts:
        raise AnalysisError("No prediction artifacts were discovered.")

    reference = artifacts[0]
    for artifact in artifacts[1:]:
        validate_paired_frames(
            reference.frame,
            artifact.frame,
            context=f"{reference.artifact_id} vs {artifact.artifact_id}",
        )
    return artifacts


def _artifact_lookup(
    artifacts: Sequence[PredictionArtifact],
) -> dict[tuple[str, str, int, str], PredictionArtifact]:
    return {
        (artifact.family, artifact.condition, artifact.seed, artifact.arm): artifact
        for artifact in artifacts
    }


def build_contrasts(
    artifacts: Sequence[PredictionArtifact],
) -> tuple[ContrastSpec, ...]:
    lookup = _artifact_lookup(artifacts)
    group_keys = sorted(
        {(item.family, item.condition, item.seed) for item in artifacts}
    )
    specs: list[ContrastSpec] = []

    def add_simple(
        *,
        family: str,
        seed: int,
        kind: str,
        left: PredictionArtifact,
        right: PredictionArtifact,
        label: str,
    ) -> None:
        specs.append(
            ContrastSpec(
                contrast_id=f"{family}|seed={seed}|{label}",
                family=family,
                seed=seed,
                kind=kind,
                terms=((left.artifact_id, 1.0), (right.artifact_id, -1.0)),
                left_artifact=left.artifact_id,
                right_artifact=right.artifact_id,
            )
        )

    for family, condition, seed in group_keys:
        available = {
            artifact.arm: artifact
            for artifact in artifacts
            if (artifact.family, artifact.condition, artifact.seed)
            == (family, condition, seed)
        }
        for left_arm, right_arm in PRIMARY_ARM_CONTRASTS:
            if left_arm in available and right_arm in available:
                add_simple(
                    family=family,
                    seed=seed,
                    kind="within_condition_arm_effect",
                    left=available[left_arm],
                    right=available[right_arm],
                    label=f"{condition}|{left_arm}-minus-{right_arm}",
                )

    control_by_family = {
        "frozen": "random_frozen",
        "online": "scratch",
    }
    seeds_by_family = {
        family: sorted({item.seed for item in artifacts if item.family == family})
        for family in ("frozen", "online")
    }
    for family, control_condition in control_by_family.items():
        for seed in seeds_by_family[family]:
            pretrained_arms = {
                item.arm: item
                for item in artifacts
                if (item.family, item.condition, item.seed)
                == (family, "panopticon_pretrained", seed)
            }
            control_arms = {
                item.arm: item
                for item in artifacts
                if (item.family, item.condition, item.seed)
                == (family, control_condition, seed)
            }
            for arm in sorted(set(pretrained_arms) & set(control_arms)):
                add_simple(
                    family=family,
                    seed=seed,
                    kind="pretraining_effect_within_arm",
                    left=pretrained_arms[arm],
                    right=control_arms[arm],
                    label=(
                        f"{arm}|panopticon_pretrained-minus-{control_condition}"
                    ),
                )

            for alternative in (
                "transient_query",
                "scale_aware_transient_query",
            ):
                required = {"current_only", alternative}
                if required.issubset(pretrained_arms) and required.issubset(
                    control_arms
                ):
                    left_alt = pretrained_arms[alternative]
                    left_current = pretrained_arms["current_only"]
                    right_alt = control_arms[alternative]
                    right_current = control_arms["current_only"]
                    specs.append(
                        ContrastSpec(
                            contrast_id=(
                                f"{family}|seed={seed}|interaction|{alternative}"
                            ),
                            family=family,
                            seed=seed,
                            kind="pretraining_by_temporal_interaction",
                            terms=(
                                (left_alt.artifact_id, 1.0),
                                (left_current.artifact_id, -1.0),
                                (right_alt.artifact_id, -1.0),
                                (right_current.artifact_id, 1.0),
                            ),
                        )
                    )

    # Aggregate frozen contrasts are linear combinations of per-seed *metrics*.
    # On every shared event-bootstrap draw we first obtain each seed's metric
    # delta, then take their equal-weight mean.  In particular, probabilities
    # are never averaged across seeds before a nonlinear metric is evaluated.
    frozen_seeds = sorted(
        {
            item.seed
            for item in artifacts
            if item.family == "frozen"
            and item.condition == "panopticon_pretrained"
        }
    )
    if len(frozen_seeds) > 1:
        aggregate_method = "equal_mean_of_seed_metric_deltas"
        seed_weight = 1.0 / float(len(frozen_seeds))

        def require_frozen(
            condition: str, seed: int, arm: str
        ) -> PredictionArtifact:
            key = ("frozen", condition, seed, arm)
            if key not in lookup:
                raise AnalysisError(
                    "Cannot construct complete frozen aggregate contrast; "
                    f"missing artifact {key}."
                )
            return lookup[key]

        for condition in sorted(EXPECTED_FROZEN_CONDITIONS):
            for left_arm, right_arm in PRIMARY_ARM_CONTRASTS:
                left_presence = [
                    ("frozen", condition, seed, left_arm) in lookup
                    for seed in frozen_seeds
                ]
                right_presence = [
                    ("frozen", condition, seed, right_arm) in lookup
                    for seed in frozen_seeds
                ]
                if not any(left_presence) or not any(right_presence):
                    continue
                if not (all(left_presence) and all(right_presence)):
                    raise AnalysisError(
                        "Frozen aggregate arm coverage is incomplete for "
                        f"{condition}: {left_arm} vs {right_arm}."
                    )
                terms: list[tuple[str, float]] = []
                for frozen_seed in frozen_seeds:
                    terms.extend(
                        (
                            (
                                require_frozen(
                                    condition, frozen_seed, left_arm
                                ).artifact_id,
                                seed_weight,
                            ),
                            (
                                require_frozen(
                                    condition, frozen_seed, right_arm
                                ).artifact_id,
                                -seed_weight,
                            ),
                        )
                    )
                specs.append(
                    ContrastSpec(
                        contrast_id=(
                            "frozen|seed=aggregate|"
                            f"{condition}|{left_arm}-minus-{right_arm}"
                        ),
                        family="frozen",
                        seed=-1,
                        kind="aggregate_within_condition_arm_effect",
                        terms=tuple(terms),
                        aggregate_method=aggregate_method,
                        aggregate_seed_count=len(frozen_seeds),
                    )
                )

        for alternative in ("transient_query", "scale_aware_transient_query"):
            required_arms = ("current_only", alternative)
            complete = all(
                ("frozen", condition, frozen_seed, arm) in lookup
                for condition in ("panopticon_pretrained", "random_frozen")
                for frozen_seed in frozen_seeds
                for arm in required_arms
            )
            any_present = any(
                ("frozen", condition, frozen_seed, alternative) in lookup
                for condition in ("panopticon_pretrained", "random_frozen")
                for frozen_seed in frozen_seeds
            )
            if not any_present:
                continue
            if not complete:
                raise AnalysisError(
                    "Frozen aggregate interaction coverage is incomplete for "
                    f"{alternative}."
                )
            terms = []
            for frozen_seed in frozen_seeds:
                terms.extend(
                    (
                        (
                            require_frozen(
                                "panopticon_pretrained",
                                frozen_seed,
                                alternative,
                            ).artifact_id,
                            seed_weight,
                        ),
                        (
                            require_frozen(
                                "panopticon_pretrained",
                                frozen_seed,
                                "current_only",
                            ).artifact_id,
                            -seed_weight,
                        ),
                        (
                            require_frozen(
                                "random_frozen",
                                frozen_seed,
                                alternative,
                            ).artifact_id,
                            -seed_weight,
                        ),
                        (
                            require_frozen(
                                "random_frozen",
                                frozen_seed,
                                "current_only",
                            ).artifact_id,
                            seed_weight,
                        ),
                    )
                )
            specs.append(
                ContrastSpec(
                    contrast_id=(
                        "frozen|seed=aggregate|interaction|"
                        f"{alternative}"
                    ),
                    family="frozen",
                    seed=-1,
                    kind=(
                        "aggregate_pretraining_by_temporal_interaction"
                    ),
                    terms=tuple(terms),
                    aggregate_method=aggregate_method,
                    aggregate_seed_count=len(frozen_seeds),
                )
            )

    identifiers = [spec.contrast_id for spec in specs]
    if not specs or len(identifiers) != len(set(identifiers)):
        raise AnalysisError("Contrast construction produced none or duplicate IDs.")
    # Every referenced artifact must exist.
    known = {artifact.artifact_id for artifact in artifacts}
    unknown = {
        artifact_id
        for spec in specs
        for artifact_id, _ in spec.terms
        if artifact_id not in known
    }
    if unknown:
        raise AnalysisError(f"Contrasts reference unknown artifacts: {sorted(unknown)}.")
    return tuple(specs)


def _event_row_groups(frame: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    event_ids = frame["event_id"].astype(str).to_numpy()
    names = sorted(set(event_ids.tolist()))
    groups = [np.flatnonzero(event_ids == event) for event in names]
    if not names or any(group.size == 0 for group in groups):
        raise AnalysisError("Canonical event grouping is empty or malformed.")
    return names, groups


def paired_event_bootstrap(
    artifacts: Sequence[PredictionArtifact],
    contrasts: Sequence[ContrastSpec],
    *,
    repeats: int,
    seed: int,
    confidence: float = 0.95,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if repeats < MIN_BOOTSTRAP_REPEATS:
        raise AnalysisError(
            f"Formal bootstrap requires at least {MIN_BOOTSTRAP_REPEATS} repeats."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1).")
    reference = artifacts[0].frame
    for artifact in artifacts[1:]:
        validate_paired_frames(
            reference,
            artifact.frame,
            context=f"bootstrap pairing with {artifact.artifact_id}",
        )
    labels = reference["label"].to_numpy(dtype=np.int8)
    if not np.array_equal(np.unique(labels), np.asarray([0, 1], dtype=np.int8)):
        raise AnalysisError("Reference predictions must contain both binary classes.")
    event_names, event_groups = _event_row_groups(reference)
    probabilities = np.stack(
        [
            artifact.frame["probability"].to_numpy(dtype=np.float64)
            for artifact in artifacts
        ],
        axis=0,
    )
    artifact_index = {
        artifact.artifact_id: index for index, artifact in enumerate(artifacts)
    }
    point_metrics = np.stack(
        [
            _binary_metrics_arrays(labels, probabilities[index])
            for index in range(len(artifacts))
        ],
        axis=0,
    )

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(
        (int(repeats), len(artifacts), len(METRIC_NAMES)), dtype=np.float64
    )
    accepted = attempted = 0
    maximum_attempts = int(repeats) * 20
    event_count = len(event_names)
    while accepted < repeats and attempted < maximum_attempts:
        attempted += 1
        sampled_events = rng.integers(0, event_count, size=event_count)
        row_indices = np.concatenate(
            [event_groups[index] for index in sampled_events]
        )
        sampled_labels = labels[row_indices]
        if np.unique(sampled_labels).size != 2:
            continue
        for artifact_position in range(len(artifacts)):
            bootstrap[accepted, artifact_position] = _binary_metrics_arrays(
                sampled_labels,
                probabilities[artifact_position, row_indices],
            )
        accepted += 1
        if progress and (accepted % 250 == 0 or accepted == repeats):
            print(
                f"[event-bootstrap] accepted={accepted}/{repeats} "
                f"attempted={attempted}",
                flush=True,
            )
    if accepted != repeats:
        raise AnalysisError(
            f"Only {accepted}/{repeats} valid two-class event bootstrap draws "
            f"after {attempted} attempts."
        )

    alpha = (1.0 - float(confidence)) / 2.0
    rows: list[dict[str, Any]] = []
    for spec in contrasts:
        coefficients = np.zeros(len(artifacts), dtype=np.float64)
        for artifact_id, coefficient in spec.terms:
            coefficients[artifact_index[artifact_id]] += float(coefficient)
        point_delta = np.einsum("am,a->m", point_metrics, coefficients)
        bootstrap_delta = np.einsum("ram,a->rm", bootstrap, coefficients)
        for metric_index, metric in enumerate(METRIC_NAMES):
            values = bootstrap_delta[:, metric_index]
            ci_low, ci_high = np.quantile(
                values, [alpha, 1.0 - alpha], method="linear"
            )
            probability_gt = float(np.mean(values > 0.0))
            probability_lt = float(np.mean(values < 0.0))
            p_le = (float(np.sum(values <= 0.0)) + 1.0) / (repeats + 1.0)
            p_ge = (float(np.sum(values >= 0.0)) + 1.0) / (repeats + 1.0)
            row: dict[str, Any] = {
                "contrast_id": spec.contrast_id,
                "family": spec.family,
                "seed": int(spec.seed),
                "kind": spec.kind,
                "metric": metric,
                "formula": spec.formula,
                "aggregate_method": spec.aggregate_method,
                "aggregate_seed_count": spec.aggregate_seed_count,
                "point_delta": float(point_delta[metric_index]),
                "bootstrap_mean_delta": float(np.mean(values)),
                "bootstrap_median_delta": float(np.median(values)),
                "bootstrap_std_delta": float(np.std(values, ddof=1)),
                "ci_level": float(confidence),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                "probability_delta_gt_0": probability_gt,
                "probability_delta_lt_0": probability_lt,
                "two_sided_bootstrap_sign_p": float(
                    min(1.0, 2.0 * min(p_le, p_ge))
                ),
                "bootstrap_repeats": int(repeats),
                "bootstrap_seed": int(seed),
                "paired_rows": int(len(reference)),
                "canonical_events": int(event_count),
                "left_artifact": spec.left_artifact,
                "right_artifact": spec.right_artifact,
            }
            if spec.left_artifact is not None and spec.right_artifact is not None:
                row["left_point_metric"] = float(
                    point_metrics[
                        artifact_index[spec.left_artifact], metric_index
                    ]
                )
                row["right_point_metric"] = float(
                    point_metrics[
                        artifact_index[spec.right_artifact], metric_index
                    ]
                )
            else:
                row["left_point_metric"] = None
                row["right_point_metric"] = None
            rows.append(row)

    group_sizes = np.asarray([group.size for group in event_groups], dtype=np.int64)
    audit = {
        "sampling_unit": "canonical_event_cluster",
        "canonical_event_rule": CANONICAL_EVENT_RULE,
        "paired_rows": int(len(reference)),
        "canonical_events": int(event_count),
        "event_rows_min": int(group_sizes.min()),
        "event_rows_max": int(group_sizes.max()),
        "event_rows_mean": float(group_sizes.mean()),
        "requested_repeats": int(repeats),
        "accepted_repeats": int(accepted),
        "attempted_draws": int(attempted),
        "seed": int(seed),
        "confidence": float(confidence),
        "shared_draws_across_all_artifacts_and_contrasts": True,
        "resampling_description": (
            "sample canonical events with replacement; concatenate every paired "
            "row belonging to each sampled event, retaining multiplicity"
        ),
    }
    return rows, audit


def _artifact_point_rows(
    artifacts: Sequence[PredictionArtifact],
) -> list[dict[str, Any]]:
    rows = []
    for artifact in artifacts:
        frame = artifact.frame
        row: dict[str, Any] = {
            "artifact_id": artifact.artifact_id,
            "family": artifact.family,
            "condition": artifact.condition,
            "seed": int(artifact.seed),
            "arm": artifact.arm,
            "epoch": int(artifact.epoch),
            "rows": int(len(frame)),
            "canonical_events": int(frame["event_id"].nunique()),
            "summary_path": str(artifact.summary_path),
            "summary_sha256": artifact.summary_sha256,
            "prediction_path": str(artifact.prediction_path),
            "prediction_sha256": artifact.prediction_sha256,
        }
        row.update({name: float(value) for name, value in artifact.point_metrics.items()})
        rows.append(row)
    return rows


def analyze(
    *,
    frozen_summaries: Sequence[os.PathLike[str] | str],
    online_summaries: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    repeats: int = MIN_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    overwrite: bool = False,
    progress: bool = False,
) -> dict[str, Path]:
    if len(frozen_summaries) != 2 or len(online_summaries) != 2:
        raise AnalysisError("Exactly two frozen and two online summaries are required.")
    frozen = tuple(load_frozen_summary(path) for path in frozen_summaries)
    online = tuple(load_online_summary(path) for path in online_summaries)
    artifacts = _validate_bundles(frozen, online)
    contrasts = build_contrasts(artifacts)
    result_rows, bootstrap_audit = paired_event_bootstrap(
        artifacts,
        contrasts,
        repeats=int(repeats),
        seed=int(seed),
        confidence=float(confidence),
        progress=progress,
    )

    destination = assert_safe_path(output_dir, purpose="analysis output directory")
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "paired_bootstrap_results.json",
        "bootstrap_csv": destination / "paired_bootstrap_results.csv",
        "artifact_csv": destination / "artifact_point_metrics.csv",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing analysis outputs: {existing}"
        )

    artifact_rows = _artifact_point_rows(artifacts)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": list(METRIC_NAMES),
        "bootstrap": bootstrap_audit,
        "input_summaries": [
            {
                "family": bundle.family,
                "condition": bundle.condition,
                "path": str(bundle.path),
                "sha256": bundle.sha256,
                "validation_manifest_sha256": bundle.validation_manifest_sha256,
                "validation_rows": int(bundle.validation_rows),
            }
            for bundle in (*frozen, *online)
        ],
        "artifacts": artifact_rows,
        "contrasts": result_rows,
        "safety": {
            "external_test_manifest_read": False,
            "manifest_files_opened": False,
            "opened_input_types": [
                "experiment_summary_json",
                "validation_best_ap_predictions_csv",
            ],
            "forbidden_path_substrings": list(BANNED_PATH_SUBSTRINGS),
            "all_prediction_rows_labels_events_strictly_paired": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    _atomic_json_write(paths["json"], payload)
    _atomic_csv_write(paths["bootstrap_csv"], pd.DataFrame(result_rows))
    _atomic_csv_write(paths["artifact_csv"], pd.DataFrame(artifact_rows))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-summaries",
        nargs=2,
        required=True,
        metavar=("PANOPTICON_SUMMARY", "RANDOM_FROZEN_SUMMARY"),
        help="The two completed heads_v2 summary.json files.",
    )
    parser.add_argument(
        "--online-summaries",
        nargs=2,
        required=True,
        metavar=("PANOPTICON_SUMMARY", "SCRATCH_SUMMARY"),
        help="The two completed online summary.json files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--repeats", type=int, default=MIN_BOOTSTRAP_REPEATS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    paths = analyze(
        frozen_summaries=args.frozen_summaries,
        online_summaries=args.online_summaries,
        output_dir=args.output_dir,
        repeats=args.repeats,
        seed=args.seed,
        confidence=args.confidence,
        overwrite=args.overwrite,
        progress=not args.quiet,
    )
    print(
        json.dumps({name: str(path) for name, path in paths.items()}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
