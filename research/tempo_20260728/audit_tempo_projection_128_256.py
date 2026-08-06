#!/usr/bin/env python3
"""Audit matched 128-D, independent-256-D, and strict-nested-256-D controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

import tempo_l89_global as global_tempo
import tempo_l89_patch as tempo


FORMAT_VERSION = "tempo-l89-projection-128-256-control-audit-v1"


def _manifest_record(
    root: Path,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "cache" / split / "manifest.json"
    manifest = tempo.load_manifest(path, expected_split=split)
    shard_bytes = sum(
        (path.parent / str(record["file"])).stat().st_size
        for record in manifest["shards"]
    )
    provenance = manifest["provenance"]
    return manifest, {
        "path": str(path),
        "sha256": tempo.sha256_file(path),
        "rows": int(manifest["rows"]),
        "shards": len(manifest["shards"]),
        "actual_shard_file_bytes": int(shard_bytes),
        "extraction_elapsed_seconds": float(
            provenance["elapsed_seconds"]
        ),
        "cuda_max_memory_allocated_bytes": int(
            provenance["cuda_max_memory_allocated_bytes"]
        ),
        "projection": manifest["configuration"]["projection"],
        "raw_768d_patch_tokens_persisted": False,
    }


def _prediction_metrics(
    path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tempo.assert_development_path(path, purpose="projection prediction")
    frame = pd.read_csv(path, low_memory=False)
    required = ("id", "plume_id", "event_id", "label", "probability")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing prediction columns {missing}")
    identity = manifest["identity"]
    expected = {
        "id": [str(value) for value in identity["ids"]],
        "plume_id": [str(value) for value in identity["plume_ids"]],
        "event_id": [str(value) for value in identity["event_ids"]],
        "label": [int(value) for value in identity["labels"]],
    }
    observed = {
        "id": frame["id"].astype(str).tolist(),
        "plume_id": frame["plume_id"].astype(str).tolist(),
        "event_id": frame["event_id"].astype(str).tolist(),
        "label": frame["label"].astype(int).tolist(),
    }
    for column in expected:
        if observed[column] != expected[column]:
            raise ValueError(f"{path}: ordered {column} differs")
    probability = frame["probability"].to_numpy(dtype=np.float64)
    if probability.shape != (int(manifest["rows"]),):
        raise ValueError(f"{path}: probability row count differs")
    if not np.isfinite(probability).all():
        raise ValueError(f"{path}: non-finite probability")
    metrics = global_tempo.metric_bundle(
        np.asarray(expected["label"], dtype=np.int64),
        probability,
        expected["event_id"],
    )
    return metrics, {
        "path": str(path),
        "sha256": tempo.sha256_file(path),
        "ordered_identity_exact": True,
    }


def _result_record(
    result_path: Path,
    prediction_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    tempo.assert_development_path(result_path, purpose="projection result")
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete":
        raise ValueError(f"{result_path}: result is not complete")
    if bool(result.get("test_or_sealed_read", True)):
        raise ValueError(f"{result_path}: held-out read contract failed")
    metrics, prediction_source = _prediction_metrics(
        prediction_path, manifest
    )
    best = result["best"]
    saved = best["validation"]
    for key in (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_macro_f1_selected",
    ):
        if abs(float(saved[key]) - float(metrics[key])) > 1e-10:
            raise ValueError(f"{result_path}: recomputed {key} differs")
    return {
        "result": str(result_path),
        "result_sha256": tempo.sha256_file(result_path),
        "prediction": prediction_source,
        "seed": int(result["configuration"]["seed"]),
        "feature_dim": int(result["configuration"]["feature_dim"]),
        "best_epoch": int(best["epoch"]),
        "epochs_completed": int(result["epochs_completed"]),
        "epoch_loop_elapsed_seconds": float(best["elapsed_seconds"]),
        "metrics_recomputed_unified_macro_threshold": metrics,
        "legacy_positive_f1_threshold_fp_mass": float(
            saved["all_negative_event_fp"]["false_positive_mass"]
        ),
    }


def _head_parameters(feature_dim: int) -> dict[str, int]:
    model = tempo.TempoPatchHead(
        feature_dim,
        match_rank=16,
        value_dim=32,
        hidden_dim=64,
        radius=1,
        temperature=0.10,
        topk_fraction=0.10,
        normality_scale=1.0,
        use_normality_features=True,
        residual_cap=1.5,
        zero_init_mode="readout",
    )
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "active": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _metric_summary(record: Mapping[str, Any]) -> dict[str, float]:
    metrics = record["metrics_recomputed_unified_macro_threshold"]
    return {
        "event_balanced_row_ap": float(metrics["event_balanced_ap"]),
        "event_balanced_row_macro_f1": float(
            metrics["event_balanced_macro_f1_selected"]
        ),
        "event_balanced_row_positive_f1": float(
            metrics["event_balanced_positive_f1_selected"]
        ),
        "event_balanced_row_auc": float(metrics["event_balanced_auc"]),
        "all_negative_fp_mass": float(metrics["all_negative_fp_mass"]),
        "all_negative_events_with_any_fp": int(
            metrics["all_negative_events_with_any_fp"]
        ),
        "all_negative_fp_rows": int(metrics["all_negative_fp_rows"]),
        "selected_macro_f1_threshold": float(
            metrics["selected_threshold"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-128", required=True)
    parser.add_argument("--root-256-independent", required=True)
    parser.add_argument("--root-256-nested", required=True)
    parser.add_argument("--result-128", required=True)
    parser.add_argument("--result-256-independent", required=True)
    parser.add_argument("--result-256-nested", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = {
        "dim128": Path(args.root_128).expanduser().resolve(),
        "dim256_independent": (
            Path(args.root_256_independent).expanduser().resolve()
        ),
        "dim256_nested": Path(args.root_256_nested).expanduser().resolve(),
    }
    result_paths = {
        "dim128": Path(args.result_128).expanduser().resolve(),
        "dim256_independent": (
            Path(args.result_256_independent).expanduser().resolve()
        ),
        "dim256_nested": Path(args.result_256_nested).expanduser().resolve(),
    }
    output = Path(args.output).expanduser().resolve()
    for name, path in {**roots, "output": output}.items():
        tempo.assert_development_path(path, purpose=name)

    manifests: dict[tuple[str, str], dict[str, Any]] = {}
    cache_records: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        cache_records[name] = {}
        for split in ("train", "val"):
            manifest, record = _manifest_record(root, split)
            manifests[(name, split)] = manifest
            cache_records[name][split] = record
        tempo.validate_cache_pair(
            root / "cache" / "train" / "manifest.json",
            root / "cache" / "val" / "manifest.json",
        )

    for split in ("train", "val"):
        reference = manifests[("dim128", split)]
        for name in ("dim256_independent", "dim256_nested"):
            candidate = manifests[(name, split)]
            if reference["identity"] != candidate["identity"]:
                raise ValueError(f"{split}: ordered identity differs for {name}")
            for key in (
                "weights_sha256",
                "base_head_sha256",
                "timepoints",
                "t0_index",
                "comparable_input_contract_sha256",
                "projection_storage_dtype",
                "amp_dtype",
            ):
                if (
                    reference["configuration"][key]
                    != candidate["configuration"][key]
                ):
                    raise ValueError(
                        f"{split}: input contract {key} differs for {name}"
                    )

    base = tempo.deterministic_orthogonal_projection(768, 128, seed=36064)
    independent = tempo.deterministic_orthogonal_projection(
        768, 256, seed=36064
    )
    nested = tempo.nested_extend_orthogonal_projection(
        768,
        256,
        base_dim=128,
        base_seed=36064,
        extension_seed=36192,
    )
    nested_prefix_equal = bool(torch.equal(nested[:, :128], base))
    if not nested_prefix_equal:
        raise AssertionError("strict nested projection prefix differs")
    independent_difference = (independent[:, :128] - base).abs()
    nesting_receipt = {
        "base_128_sha256": tempo.tensor_sha256(base),
        "independent_256_sha256": tempo.tensor_sha256(independent),
        "nested_256_sha256": tempo.tensor_sha256(nested),
        "nested_first_128_array_equal": nested_prefix_equal,
        "nested_first_128_max_abs_error": float(
            (nested[:, :128] - base).abs().max()
        ),
        "nested_max_abs_qtq_minus_i": float(
            (nested.T @ nested - torch.eye(256)).abs().max()
        ),
        "independent_first_128_max_abs_difference": float(
            independent_difference.max()
        ),
        "independent_first_128_mean_abs_difference": float(
            independent_difference.mean()
        ),
    }

    result_records: dict[str, dict[str, Any]] = {}
    for name, result_path in result_paths.items():
        prediction_path = result_path.parent / "validation_predictions.csv"
        result_records[name] = _result_record(
            result_path,
            prediction_path,
            manifests[(name, "val")],
        )
    if {record["seed"] for record in result_records.values()} != {20260728}:
        raise ValueError("projection controls do not share seed 20260728")

    summaries = {
        name: _metric_summary(record)
        for name, record in result_records.items()
    }
    base_summary = summaries["dim128"]
    required_ap = base_summary["event_balanced_row_ap"] + 0.002
    required_macro = base_summary["event_balanced_row_macro_f1"]
    comparisons: dict[str, dict[str, Any]] = {}
    for name in ("dim256_independent", "dim256_nested"):
        summary = summaries[name]
        comparisons[name] = {
            "delta_vs_matched_128": {
                key: float(summary[key] - base_summary[key])
                for key in (
                    "event_balanced_row_ap",
                    "event_balanced_row_macro_f1",
                    "event_balanced_row_positive_f1",
                    "event_balanced_row_auc",
                    "all_negative_fp_mass",
                )
            },
            "single_seed_gate": {
                "required_ap": required_ap,
                "required_macro_f1": required_macro,
                "ap_pass": (
                    summary["event_balanced_row_ap"] >= required_ap
                ),
                "macro_f1_pass": (
                    summary["event_balanced_row_macro_f1"]
                    >= required_macro
                ),
            },
        }
        comparisons[name]["single_seed_gate"]["overall_pass"] = bool(
            comparisons[name]["single_seed_gate"]["ap_pass"]
            and comparisons[name]["single_seed_gate"]["macro_f1_pass"]
        )

    audit = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "same_ordered_identity_and_input_contract": True,
        "test_or_sealed_read": False,
        "projection_nesting_receipt": nesting_receipt,
        "caches": cache_records,
        "head_parameters": {
            "dim128_readout_zero": _head_parameters(128),
            "dim256_readout_zero": _head_parameters(256),
        },
        "results": result_records,
        "metric_summaries": summaries,
        "comparisons": comparisons,
        "verdict": {
            "independent_256": (
                "passes the numerical single-seed screen but is an "
                "independent random-subspace sensitivity result; it is not a "
                "capacity intervention and is not expanded to more seeds"
            ),
            "strict_nested_256": (
                "fails the numerical single-seed screen and regresses versus "
                "the bit-compatible 128-D seed; stop without more seeds"
            ),
            "capacity_claim_supported": False,
        },
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": tempo.sha256_file(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tempo.atomic_json(output, audit)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": tempo.sha256_file(output),
                "comparisons": comparisons,
                "verdict": audit["verdict"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
