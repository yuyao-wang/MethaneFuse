#!/usr/bin/env python3
"""Collect development, gate, and sealed metrics into one protocol-aware registry.

This collector only reads JSON metric artifacts.  It never opens a manifest,
prediction file, checkpoint, or source image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


VALIDATION_METRICS = (
    ("test_f1", "validation_best_f1_at_0p5"),
    ("test_macro_f1", "validation_best_macro_f1_at_0p5"),
    ("test_best_f1", "validation_best_threshold_optimized_f1"),
    ("test_auroc", "validation_best_auroc"),
    ("test_ap", "validation_best_ap"),
    ("test_recall_at_fpr_01", "validation_best_recall_at_fpr_01"),
    ("test_recall_at_fpr_05", "validation_best_recall_at_fpr_05"),
)

SEALED_METRICS = (
    "threshold",
    "accuracy",
    "f1",
    "macro_f1",
    "recall",
    "fpr",
    "auroc",
    "ap",
    "recall_at_fpr_01",
    "recall_at_fpr_05",
    "tp",
    "fp",
    "fn",
    "tn",
    "samples",
    "positives",
    "negatives",
)

GATE_METRICS = ("ap", "auroc", "best_f1", "f1_at_0p5")
GATE_COMPARISONS_BY_ARTIFACT_TYPE = {
    "mae_scratch_gate_decision": (
        "scratch",
        "finetune",
        "delta_finetune_minus_scratch",
    ),
    "validity_masked_mae_scratch_gate_decision": (
        "scratch",
        "finetune",
        "delta_finetune_minus_scratch",
    ),
    "ranknet_bce_scratch_gate_decision": (
        "scratch",
        "ranknet",
        "delta_ranknet_minus_scratch",
    ),
}
GATE_ARTIFACT_TYPES = set(GATE_COMPARISONS_BY_ARTIFACT_TYPE)


def finite_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def best_record(history: list[dict], metric: str) -> dict:
    candidates = [record for record in history if finite_number(record.get(metric))]
    return max(candidates, key=lambda record: float(record[metric])) if candidates else {}


def _base_row(
    path: Path, *, run_name: str | None = None, group: str | None = None
) -> dict:
    return {
        "run_name": run_name or path.parent.name,
        "group": group or path.parent.parent.name,
        "metrics_path": str(path),
        "updated_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _truthy_marker(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def protocol_exclusion_reasons(
    path: Path, artifact: Any, *, check_sidecar: bool = True
) -> list[str]:
    """Return explicit reasons why an artifact must not enter the registry."""

    reasons: list[str] = []
    if check_sidecar and (path.parent / "PROTOCOL_INVALID.json").is_file():
        reasons.append("PROTOCOL_INVALID.json sidecar exists")

    records: list[dict] = []
    if isinstance(artifact, dict):
        records.append(artifact)
        epochs = artifact.get("epochs")
        if isinstance(epochs, list):
            records.extend(record for record in epochs if isinstance(record, dict))
    elif isinstance(artifact, list):
        records.extend(record for record in artifact if isinstance(record, dict))

    if any(
        record.get("protocol_status") == "invalid_diagnostic_only"
        for record in records
    ):
        reasons.append("protocol_status=invalid_diagnostic_only")
    if any(
        _truthy_marker(record.get("do_not_use_for_model_selection"))
        for record in records
    ):
        reasons.append("do_not_use_for_model_selection=true")
    if any(record.get("protocol_valid") is False for record in records):
        reasons.append("protocol_valid=false")
    return reasons


def collect_metric_history(path: Path, history: Any) -> dict:
    if isinstance(history, dict):
        epoch_records = history.get("epochs", [])
        if not isinstance(epoch_records, list):
            raise ValueError(f"Expected an epoch list in structured history: {path}")
        config = history.get("config", {})
        config = config if isinstance(config, dict) else {}
        event_protocol = history.get("event_protocol", {})
        event_protocol = event_protocol if isinstance(event_protocol, dict) else {}
        data_signature = history.get("data_signature", {})
        data_signature = data_signature if isinstance(data_signature, dict) else {}
        policy = config.get("cross_sensor_event_policy")
        overlap_after = event_protocol.get("global_train_val_overlap_after")
        if overlap_after is None:
            overlap_after = data_signature.get("global_train_val_overlap_after")
        fingerprint = event_protocol.get("fingerprint")
        if fingerprint is None:
            fingerprint = data_signature.get("event_protocol_fingerprint")
        evaluation_split = (
            "pretraining" if history.get("mode") == "pretrain" else "validation"
        )
        init_audit = history.get("init_checkpoint", {})
        init_audit = init_audit if isinstance(init_audit, dict) else {}
        row = {
            **_base_row(path),
            "record_format": "multisensor",
            "evaluation_split": evaluation_split,
            "status": history.get("status"),
            "mode": history.get("mode"),
            "sharing": history.get("sharing"),
            "model_parameters": history.get("model_parameters"),
            "epochs_recorded": len(epoch_records),
            "last_epoch": epoch_records[-1].get("epoch") if epoch_records else None,
            "last_global_step": None,
            "protocol_status": history.get("protocol_status"),
            "cross_sensor_event_policy": policy,
            "protocol_manifest_dir": config.get("protocol_manifest_dir"),
            "event_protocol_fingerprint": fingerprint,
            "global_train_val_overlap_before": event_protocol.get(
                "global_train_val_overlap_before"
            ),
            "global_train_val_overlap_after": overlap_after,
            "global_purge_verified": policy == "purge" and overlap_after == 0,
            "init_checkpoint": config.get("init_checkpoint"),
            "init_checkpoint_sha256": init_audit.get("checkpoint_sha256"),
            "init_encoder_coverage": init_audit.get("encoder_coverage"),
            "initial_head_sha256": history.get("initial_head_sha256"),
        }
        candidates = [
            record
            for record in epoch_records
            if isinstance(record, dict)
            and isinstance(record.get("macro_over_sensor"), dict)
        ]
        if candidates:
            if evaluation_split == "pretraining":
                reconstruction_candidates = [
                    record
                    for record in candidates
                    if finite_number(
                        record["macro_over_sensor"].get("reconstruction_loss")
                    )
                ]
                selected = (
                    min(
                        reconstruction_candidates,
                        key=lambda record: float(
                            record["macro_over_sensor"]["reconstruction_loss"]
                        ),
                    )
                    if reconstruction_candidates
                    else candidates[-1]
                )
                row.update(
                    {
                        "pretraining_selected_epoch": selected.get("epoch"),
                        "pretraining_epoch_wall_time_seconds": selected.get(
                            "elapsed_seconds"
                        ),
                        "pretraining_macro_reconstruction_loss": selected[
                            "macro_over_sensor"
                        ].get("reconstruction_loss"),
                        "pretraining_sensor_reconstruction_json": json.dumps(
                            selected.get("val", {}), sort_keys=True
                        ),
                    }
                )
                return row
            ap_candidates = [
                record
                for record in candidates
                if finite_number(record["macro_over_sensor"].get("ap"))
            ]
            selected = (
                max(
                    ap_candidates,
                    key=lambda record: float(record["macro_over_sensor"]["ap"]),
                )
                if ap_candidates
                else candidates[-1]
            )
            macro = selected["macro_over_sensor"]
            row.update(
                {
                    "validation_macro_selected_epoch": selected.get("epoch"),
                    "validation_epoch_wall_time_seconds": selected.get(
                        "elapsed_seconds"
                    ),
                    "validation_macro_f1_at_0p5": macro.get("f1_0p5"),
                    "validation_macro_threshold_optimized_f1": macro.get("f1"),
                    "validation_macro_ap": macro.get("ap"),
                    "validation_macro_auroc": macro.get("auroc"),
                    "validation_sensor_metrics_json": json.dumps(
                        selected.get("val", {}), sort_keys=True
                    ),
                }
            )
        return row

    if not isinstance(history, list) or not history:
        raise ValueError(f"Expected a nonempty metric-history list: {path}")
    row = {
        **_base_row(path),
        "record_format": "sensor",
        "evaluation_split": "validation",
        "epochs_recorded": len(history),
        "last_epoch": history[-1].get("epoch"),
        "last_global_step": history[-1].get("global_step"),
    }
    for source_metric, registry_metric in VALIDATION_METRICS:
        record = best_record(history, source_metric)
        row[registry_metric] = record.get(source_metric)
        row[f"{registry_metric}_epoch"] = record.get("epoch")
    ap_record = best_record(history, "test_ap")
    row["validation_ap_selected_f1_at_0p5"] = ap_record.get("test_f1")
    row["validation_ap_selected_threshold_optimized_f1"] = ap_record.get(
        "test_best_f1"
    )
    row["validation_ap_selected_threshold"] = ap_record.get(
        "test_best_f1_threshold"
    )
    return row


def _infer_sealed_sensor(path: Path) -> str:
    run_name = path.stem
    suffix = "_sealed_test"
    prefix = run_name[: -len(suffix)] if run_name.endswith(suffix) else run_name
    return prefix.split("_", 1)[0].lower()


def collect_sealed_result(path: Path, artifact: Any) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError(f"Expected a structured sealed-test artifact: {path}")
    artifact_type = artifact.get("artifact_type")
    if artifact_type not in (None, "sealed_test_result"):
        raise ValueError(
            f"Unexpected artifact_type={artifact_type!r} in sealed result: {path}"
        )
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a metrics object in sealed result: {path}")
    missing = [
        metric for metric in ("f1", "ap", "auroc") if not finite_number(metrics.get(metric))
    ]
    if missing:
        raise ValueError(f"Missing finite sealed metrics {missing}: {path}")

    checkpoint = artifact.get("checkpoint")
    checkpoint_path = checkpoint.get("path") if isinstance(checkpoint, dict) else checkpoint
    checkpoint_epoch = (
        checkpoint.get("epoch")
        if isinstance(checkpoint, dict)
        else artifact.get("checkpoint_epoch")
    )
    sealed_manifest = artifact.get("sealed_manifest")
    if isinstance(sealed_manifest, dict):
        sealed_manifest_path = sealed_manifest.get("path")
        sealed_manifest_sha256 = sealed_manifest.get("sha256")
        sealed_manifest_rows = sealed_manifest.get("rows")
    else:
        sealed_manifest_path = artifact.get("eval_csv")
        sealed_manifest_sha256 = None
        sealed_manifest_rows = None

    row = {
        **_base_row(path, run_name=path.stem, group="sealed_test"),
        "record_format": "sealed_test",
        "evaluation_split": "sealed_test",
        "artifact_type": artifact_type or "sealed_test_result_legacy",
        "protocol": artifact.get("protocol"),
        "sensor": artifact.get("sensor") or _infer_sealed_sensor(path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint_epoch,
        "sealed_manifest_path": sealed_manifest_path,
        "sealed_manifest_sha256": sealed_manifest_sha256,
        "sealed_manifest_rows": sealed_manifest_rows,
        "development_event_overlap": artifact.get("development_event_overlap"),
        "threshold_source_json": json.dumps(
            artifact.get("threshold_source"), sort_keys=True
        ),
    }
    for metric in SEALED_METRICS:
        row[f"sealed_{metric}"] = metrics.get(metric)
    return row


def collect_gate_result(path: Path, artifact: Any) -> dict:
    """Index a preregistered promotion decision without treating failure as invalid."""

    if not isinstance(artifact, dict):
        raise ValueError(f"Expected a structured promotion-gate artifact: {path}")
    artifact_type = artifact.get("artifact_type")
    if artifact_type not in GATE_ARTIFACT_TYPES:
        raise ValueError(
            f"Unexpected artifact_type={artifact_type!r} in promotion gate: {path}"
        )

    status = artifact.get("status")
    gate_decision = artifact.get("gate_decision")
    preregistered_gate = artifact.get("preregistered_gate")
    metrics = artifact.get("metrics")
    checks = artifact.get("checks")
    if status not in {"pass", "fail"}:
        raise ValueError(f"Expected promotion-gate status pass/fail: {path}")
    if not isinstance(gate_decision, dict):
        raise ValueError(f"Expected a gate_decision object: {path}")
    if not isinstance(gate_decision.get("pass"), bool):
        raise ValueError(f"Expected a Boolean gate_decision.pass: {path}")
    gate_pass = gate_decision["pass"]
    expected_status = "pass" if gate_pass else "fail"
    if status != expected_status:
        raise ValueError(
            f"Inconsistent status={status!r} and gate_decision.pass={gate_pass}: {path}"
        )
    if not isinstance(preregistered_gate, dict):
        raise ValueError(f"Expected a preregistered_gate object: {path}")
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a metrics object in promotion gate: {path}")
    if not isinstance(checks, dict) or not checks:
        raise ValueError(f"Expected nonempty integrity checks: {path}")

    comparisons = GATE_COMPARISONS_BY_ARTIFACT_TYPE[artifact_type]
    macro = metrics.get("macro_mean_over_sensors")
    per_sensor = metrics.get("per_sensor")
    if not isinstance(macro, dict) or not isinstance(per_sensor, dict):
        raise ValueError(f"Expected macro and per-sensor gate metrics: {path}")
    for comparison in comparisons:
        comparison_metrics = macro.get(comparison)
        if not isinstance(comparison_metrics, dict):
            raise ValueError(f"Missing macro comparison {comparison!r}: {path}")
        missing = [
            metric
            for metric in GATE_METRICS
            if not finite_number(comparison_metrics.get(metric))
        ]
        if missing:
            raise ValueError(
                f"Missing finite macro gate metrics {missing} for {comparison}: {path}"
            )
    for sensor, sensor_result in per_sensor.items():
        if not isinstance(sensor_result, dict):
            raise ValueError(f"Expected a result mapping for sensor {sensor!r}: {path}")
        for comparison in comparisons:
            comparison_metrics = sensor_result.get(comparison)
            if not isinstance(comparison_metrics, dict):
                raise ValueError(
                    f"Missing {comparison!r} metrics for sensor {sensor!r}: {path}"
                )
            missing = [
                metric
                for metric in GATE_METRICS
                if not finite_number(comparison_metrics.get(metric))
            ]
            if missing:
                raise ValueError(
                    f"Missing finite gate metrics {missing} for "
                    f"{sensor}/{comparison}: {path}"
                )

    check_results = [
        check.get("pass")
        for check in checks.values()
        if isinstance(check, dict)
    ]
    if len(check_results) != len(checks) or any(
        not isinstance(result, bool) for result in check_results
    ):
        raise ValueError(f"Every promotion-gate check must have Boolean pass: {path}")
    checks_passed = sum(check_results)
    failed_criteria = gate_decision.get("failed_criteria")
    if not isinstance(failed_criteria, list):
        raise ValueError(f"Expected gate_decision.failed_criteria list: {path}")
    criteria = gate_decision.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError(f"Expected nonempty gate_decision.criteria: {path}")
    criterion_results = [
        criterion.get("pass")
        for criterion in criteria.values()
        if isinstance(criterion, dict)
    ]
    if len(criterion_results) != len(criteria) or any(
        not isinstance(result, bool) for result in criterion_results
    ):
        raise ValueError(
            f"Every promotion-gate criterion must have Boolean pass: {path}"
        )
    criteria_passed = sum(criterion_results)

    row = {
        **_base_row(
            path,
            run_name=preregistered_gate.get("name") or path.stem,
            group="promotion_gate",
        ),
        "record_format": "promotion_gate",
        "evaluation_split": "promotion_gate",
        "artifact_type": artifact_type,
        "status": status,
        "gate_name": preregistered_gate.get("name"),
        "gate_pass": gate_pass,
        "gate_decision": "promote" if gate_pass else "do_not_promote",
        "do_not_promote": not gate_pass,
        "gate_interpretation": gate_decision.get("interpretation"),
        "gate_failed_criteria_json": json.dumps(failed_criteria, sort_keys=True),
        "gate_criteria_json": json.dumps(criteria, sort_keys=True),
        "gate_criteria_passed": criteria_passed,
        "gate_criteria_total": len(criteria),
        "gate_criteria_all_pass": criteria_passed == len(criteria),
        "gate_preregistered_definition_json": json.dumps(
            preregistered_gate, sort_keys=True
        ),
        "gate_comparison_baseline": comparisons[0],
        "gate_comparison_candidate": comparisons[1],
        "gate_comparison_delta": comparisons[2],
        "gate_integrity_checks_passed": checks_passed,
        "gate_integrity_checks_total": len(checks),
        "gate_integrity_all_checks_pass": checks_passed == len(checks),
        "gate_integrity_checks_json": json.dumps(checks, sort_keys=True),
        "gate_macro_metrics_json": json.dumps(macro, sort_keys=True),
        "gate_per_sensor_metrics_json": json.dumps(per_sensor, sort_keys=True),
        "gate_created_utc": artifact.get("created_utc"),
    }
    for comparison in comparisons:
        for metric in GATE_METRICS:
            row[f"gate_macro_{comparison}_{metric}"] = macro[comparison][metric]
    return row


def iter_metric_paths(root: Path) -> Iterator[Path]:
    """Scan experiment trees only; image caches contain millions of folders."""

    for directory_name in ("checkpoints", "shared_residual"):
        directory = root / directory_name
        if directory.is_dir():
            yield from directory.rglob("metrics_history.json")


def iter_sealed_paths(root: Path) -> Iterator[Path]:
    """Read only already-materialized sealed result JSONs, never test data."""

    results_dir = root / "results"
    if results_dir.is_dir():
        yield from results_dir.glob("*_sealed_test.json")


def iter_gate_paths(root: Path) -> Iterator[Path]:
    """Read only already-materialized gate decisions, never underlying data."""

    results_dir = root / "results"
    if results_dir.is_dir():
        yield from results_dir.glob("*_gate.json")


def _atomic_write_text(path: Path, text: str) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(
            file_descriptor, "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Audit and print counts without writing registry files.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    failures: list[dict] = []
    exclusions: list[dict] = []
    metric_paths = sorted(iter_metric_paths(args.root))
    sealed_paths = sorted(iter_sealed_paths(args.root))
    gate_paths = sorted(iter_gate_paths(args.root))
    for path in metric_paths:
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
            reasons = protocol_exclusion_reasons(path, history)
            if reasons:
                exclusions.append({"path": str(path), "reasons": reasons})
                continue
            rows.append(collect_metric_history(path, history))
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "artifact_kind": "metric_history",
                    "error": repr(exc),
                }
            )
    for path in sealed_paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            reasons = protocol_exclusion_reasons(
                path, artifact, check_sidecar=False
            )
            if reasons:
                exclusions.append({"path": str(path), "reasons": reasons})
                continue
            rows.append(collect_sealed_result(path, artifact))
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "artifact_kind": "sealed_test_result",
                    "error": repr(exc),
                }
            )
    gate_artifact_index: list[dict] = []
    for path in gate_paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            reasons = protocol_exclusion_reasons(
                path, artifact, check_sidecar=False
            )
            if reasons:
                exclusions.append({"path": str(path), "reasons": reasons})
                continue
            row = collect_gate_result(path, artifact)
            rows.append(row)
            gate_artifact_index.append(
                {
                    "path": str(path),
                    "artifact_type": row["artifact_type"],
                    "gate_name": row["gate_name"],
                    "status": row["status"],
                    "gate_pass": row["gate_pass"],
                    "gate_decision": row["gate_decision"],
                    "do_not_promote": row["do_not_promote"],
                    "comparison_baseline": row["gate_comparison_baseline"],
                    "comparison_candidate": row["gate_comparison_candidate"],
                    "comparison_delta": row["gate_comparison_delta"],
                    "integrity_checks_passed": row[
                        "gate_integrity_checks_passed"
                    ],
                    "integrity_checks_total": row[
                        "gate_integrity_checks_total"
                    ],
                    "criteria_passed": row["gate_criteria_passed"],
                    "criteria_total": row["gate_criteria_total"],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "artifact_kind": "promotion_gate_result",
                    "error": repr(exc),
                }
            )
    rows.sort(
        key=lambda row: (
            row["group"],
            row["run_name"],
            row.get("evaluation_split", ""),
        )
    )

    split_counts: dict[str, int] = {}
    for row in rows:
        split = row.get("evaluation_split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    audit = {
        "runs_collected": len(rows),
        "runs_by_evaluation_split": split_counts,
        "sources_discovered": {
            "metric_histories": len(metric_paths),
            "sealed_test_results": len(sealed_paths),
            "promotion_gate_results": len(gate_paths),
        },
        "gate_artifact_index": gate_artifact_index,
        "excluded_count": len(exclusions),
        "exclusions": exclusions,
        "failure_count": len(failures),
        "failures": failures,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if args.dry_run:
        print(json.dumps({"dry_run": True, **audit}, indent=2), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "experiment_registry.json"
    csv_path = args.output_dir / "experiment_registry.csv"
    audit_path = args.output_dir / "experiment_registry_audit.json"

    _atomic_write_text(json_path, json.dumps(rows, indent=2) + "\n")
    _atomic_write_csv(csv_path, rows)
    _atomic_write_text(audit_path, json.dumps(audit, indent=2) + "\n")
    print(
        f"[registry] runs={len(rows)} excluded={len(exclusions)} "
        f"failures={len(failures)} "
        f"json={json_path} csv={csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
