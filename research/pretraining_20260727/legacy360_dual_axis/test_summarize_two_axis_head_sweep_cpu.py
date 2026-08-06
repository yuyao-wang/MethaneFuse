from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import summarize_two_axis_head_sweep as utility


BEST_SCORES = {
    "hybrid_current_only_lr1e4": 0.84,
    "hybrid_current_only_lr3e4": 0.85,
    "hybrid_two_axis_lr1e4": 0.87,
    "hybrid_two_axis_lr3e4": 0.88,
    "hybrid_scale_aware_lr1e4": 0.89,
    "hybrid_scale_aware_lr3e4": 0.91,
    "universal_current_only_lr1e4": 0.80,
    "universal_current_only_lr3e4": 0.81,
    "universal_scale_aware_lr1e4": 0.83,
    "universal_scale_aware_lr3e4": 0.84,
}


@contextmanager
def _raises(error_type: type[BaseException], match: str):
    try:
        yield
    except error_type as error:
        assert re.search(match, str(error)), (match, str(error))
    else:
        raise AssertionError(f"expected {error_type.__name__}: {match}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _metric_leaf(score: float, threshold: float) -> dict[str, Any]:
    fixed_binary = max(0.0, score - 0.012)
    fixed_macro = max(0.0, score - 0.022)
    return {
        "rows": 100,
        "positives": 50,
        "binary_f1": fixed_binary,
        "macro_f1": fixed_macro,
        "balanced_accuracy": max(0.0, score - 0.018),
        "predicted_positive_rate": 0.48,
        "binary_f1_at_0_5": fixed_binary,
        "macro_f1_at_0_5": fixed_macro,
        "balanced_accuracy_at_0_5": max(0.0, score - 0.018),
        "predicted_positive_rate_at_0_5": 0.48,
        "decision_threshold": 0.5,
        "ap": min(1.0, score + 0.025),
        "auc": min(1.0, score + 0.045),
        "best_binary_f1": score,
        "best_binary_f1_threshold": threshold,
        "best_macro_f1_at_binary_threshold": max(0.0, score - 0.008),
    }


def _metrics(score: float, threshold: float = 0.42) -> dict[str, Any]:
    result = _metric_leaf(score, threshold)
    result["availability"] = {
        "s2+l89": _metric_leaf(max(0.0, score - 0.005), threshold)
    }
    result["by_sensor"] = {
        sensor: _metric_leaf(
            max(0.0, score - 0.004 * index), threshold
        )
        for index, sensor in enumerate(utility.SENSORS)
    }
    return result


def _make_run(root: Path, spec: utility.RunSpec) -> None:
    run_dir = root / spec.name
    run_dir.mkdir(parents=True)
    epoch0_score = 0.80 if spec.base_mode == "hybrid" else 0.75
    best_score = BEST_SCORES[spec.name]
    history = [
        {
            "epoch": 0,
            "train": None,
            "dev": _metrics(epoch0_score, 0.43),
            "elapsed_seconds": 0.0,
            "interpretation": "exact warm-start base",
        },
        {
            "epoch": 1,
            "train": {"loss": 0.5, "rows": 1000, "steps": 2},
            "dev": _metrics(best_score - 0.02, 0.41),
            "elapsed_seconds": 1.0,
        },
        {
            "epoch": 2,
            "train": {"loss": 0.4, "rows": 1000, "steps": 2},
            "dev": _metrics(best_score, 0.39),
            "elapsed_seconds": 2.0,
        },
        {
            "epoch": 3,
            "train": {"loss": 0.35, "rows": 1000, "steps": 2},
            "dev": _metrics(best_score - 0.01, 0.40),
            "elapsed_seconds": 3.0,
        },
    ]
    best = history[2]
    encoder = {"state_sha256": "e" * 64, "source": "synthetic"}
    model_config = {
        "embed_dim": 768,
        "num_sensors": 4,
        "num_roles": 3,
        "model_dim": 256,
        "num_heads": 8,
        "temporal_depth": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
    }
    lock = {
        "schema_version": utility.LOCK_SCHEMA,
        "locked_utc": "2026-07-27T00:00:00+00:00",
        "checkpoint": str((run_dir / "checkpoint_best.pth").absolute()),
        "checkpoint_sha256": "c" * 64,
        "best_epoch": 2,
        "selection_metric": utility.EXPECTED_SELECTION_METRIC,
        "selection_score": best_score,
        "locked_threshold": 0.39,
        "arm": spec.arm,
        "base_mode": spec.base_mode,
        "model_config": model_config,
        "encoder": encoder,
        "split_guard": {
            "train_plume_ids": ["train-plume"],
            "dev_plume_ids": ["dev-plume"],
            "source_train_event_ids": ["dev-event", "train-event"],
        },
        "train_manifest": {
            "path": "/synthetic/train_core.csv",
            "sha256": "a" * 64,
            "rows": 1000,
        },
        "dev_manifest": {
            "path": "/synthetic/dev.csv",
            "sha256": "b" * 64,
            "rows": 100,
        },
        "threshold_source": "development only",
        "test_cache_read_before_lock": False,
        "protocol_caveat": "synthetic CPU test",
    }
    summary = {
        "schema_version": utility.SUMMARY_SCHEMA,
        "script_version": "query360-two-axis-full-legacy-v2",
        "status": "complete",
        "protocol": "development only",
        "train_rows": 1000,
        "dev_rows": 100,
        "encoder": encoder,
        "model": {
            "arm": spec.arm,
            "base_mode": spec.base_mode,
            "initial_state_sha256": "i" * 64,
            "parameter_signature": {
                "trainable": ["fused_residual.weight"],
                "numel": 123,
            },
            "model_dim": 256,
            "num_heads": 8,
            "temporal_depth": 2,
        },
        "training": {
            "seed": 42,
            "epochs": 3,
            "batch_size": 512,
            "learning_rate": spec.learning_rate,
            "sensor_aux_weight": 0.3,
            "axis_aux_weight": 0.15,
            "selection_metric": utility.EXPECTED_SELECTION_METRIC,
        },
        "best": best,
        "selection_lock": lock,
        "sealed_test": None,
        "sealed_test_read": False,
        "sealed_test_evaluations": 0,
        "history": history,
        "metric_guardrail": "development only",
        "protocol_caveat": "synthetic CPU test",
    }
    status = {
        "status": "complete",
        "completed_utc": "2026-07-27T00:01:00+00:00",
        "best_epoch": 2,
        "best_dev_metrics": best["dev"],
        "locked_threshold": 0.39,
        "sealed_test_read": False,
        "sealed_test_evaluations": 0,
    }
    _write_json(run_dir / "metrics_history.json", history)
    _write_json(run_dir / "selection_lock.json", lock)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "run_status.json", status)


def _make_sweep(tmp_path: Path) -> Path:
    heads = tmp_path / "heads"
    heads.mkdir(parents=True)
    for spec in utility.RUN_SPECS:
        _make_run(heads, spec)
    return heads


def test_synthetic_ten_run_summary_and_outputs(tmp_path: Path) -> None:
    heads = _make_sweep(tmp_path)
    # A poisoned test-looking artifact proves that the summarizer only opens
    # the four exact development artifacts from its input contract.
    _write_json(
        heads / utility.RUN_SPECS[0].name / "test_results.json",
        {"sealed_test_read": True, "f1": 1.0},
    )

    report = utility.summarize_sweep(heads)
    assert report["run_count"] == 10
    assert report["sealed_test_read"] is False
    assert (
        report["winner_recommendation"]["recommended_run"]
        == "hybrid_scale_aware_lr3e4"
    )
    assert report["winner_recommendation"]["selection_score"] == 0.91
    assert len(report["matched_deltas_vs_current_only"]) == 6
    target_delta = next(
        delta
        for delta in report["matched_deltas_vs_current_only"]
        if delta["candidate_run"] == "hybrid_scale_aware_lr3e4"
    )
    assert target_delta["control_run"] == "hybrid_current_only_lr3e4"
    assert abs(target_delta["selection_metric_delta"] - 0.06) < 1e-12

    paths = utility.write_reports(report, tmp_path / "reports")
    assert set(paths) == {"json", "runs_csv", "deltas_csv", "markdown"}
    assert all(path.is_file() for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["winner_recommendation"]["selection_score"] == 0.91
    with paths["runs_csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert "best_s2_binary_f1_at_0_5" in rows[0]
    assert "epoch0_by_sensor_json" in rows[0]
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "No test or sealed-test input is supported" in markdown
    assert "hybrid_scale_aware_lr3e4" in markdown


def test_rejects_failed_or_sealed_run_before_writing(tmp_path: Path) -> None:
    heads = _make_sweep(tmp_path)
    failed_path = heads / utility.RUN_SPECS[0].name / "run_status.json"
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    failed["status"] = "failed"
    _write_json(failed_path, failed)
    with _raises(utility.SweepValidationError, match="not complete"):
        utility.summarize_sweep(heads)

    heads = _make_sweep(tmp_path / "second")
    summary_path = heads / utility.RUN_SPECS[1].name / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["sealed_test_read"] = True
    _write_json(summary_path, summary)
    with _raises(utility.SweepValidationError, match="exactly false"):
        utility.summarize_sweep(heads)
    assert not (tmp_path / "reports").exists()


def test_rejects_cross_artifact_and_epoch0_inconsistency(
    tmp_path: Path,
) -> None:
    heads = _make_sweep(tmp_path)
    lock_path = heads / utility.RUN_SPECS[2].name / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["locked_threshold"] = 0.99
    _write_json(lock_path, lock)
    with _raises(utility.SweepValidationError, match="embedded vs external"):
        utility.summarize_sweep(heads)

    heads = _make_sweep(tmp_path / "second")
    run = utility.RUN_SPECS[3]
    history_path = heads / run.name / "metrics_history.json"
    summary_path = heads / run.name / "summary.json"
    status_path = heads / run.name / "run_status.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history[0]["dev"] = _metrics(0.79, 0.43)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["history"] = history
    status = json.loads(status_path.read_text(encoding="utf-8"))
    # The selected epoch is unchanged, so only the exact-base cross-run audit
    # should reject this otherwise internally consistent mutation.
    _write_json(history_path, history)
    _write_json(summary_path, summary)
    _write_json(status_path, status)
    with _raises(
        utility.SweepValidationError, match="epoch-0 exact-base consistency"
    ):
        utility.summarize_sweep(heads)


def test_report_destination_cannot_modify_heads_root(tmp_path: Path) -> None:
    heads = _make_sweep(tmp_path)
    report = utility.summarize_sweep(heads)
    with _raises(utility.SweepValidationError, match="outside the heads root"):
        utility.write_reports(report, heads / "reports")
    assert not (heads / "reports").exists()


class SyntheticSummaryTests(unittest.TestCase):
    """Standard-library runner for environments without pytest installed."""

    def _run_case(self, case) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case(Path(temporary))

    def test_summary_and_outputs(self) -> None:
        self._run_case(test_synthetic_ten_run_summary_and_outputs)

    def test_failed_or_sealed(self) -> None:
        self._run_case(test_rejects_failed_or_sealed_run_before_writing)

    def test_artifact_and_base_inconsistency(self) -> None:
        self._run_case(test_rejects_cross_artifact_and_epoch0_inconsistency)

    def test_read_only_destination(self) -> None:
        self._run_case(test_report_destination_cannot_modify_heads_root)


if __name__ == "__main__":
    unittest.main()
