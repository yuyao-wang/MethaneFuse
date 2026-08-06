#!/usr/bin/env python3
"""Fixed multi-seed L89 patch/D1/P5 ensemble and event-bootstrap audit.

No ensemble weight is fitted.  The only reported combinations are:

* two or more declared patch seeds averaged in logit space;
* three D1 seeds averaged in logit space;
* P5 + D1 with fixed 1/2, 1/2 logit weights;
* P5 + D1 + patch with fixed 1/3, 1/3, 1/3 logit weights.

Every input must exactly match the ordered validation identity in the
projected-patch manifest.  This script rejects test/sealed/holdout paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import tempo_l89_patch as tempo
import tempo_l89_global as global_tempo
from research.pretraining_20260727 import l89_ragged_cls_experiment as l89


FORMAT_VERSION = "tempo-l89-patch-multiseed-fixed-audit-v1"


def _identity_frame(manifest: Mapping[str, Any]) -> pd.DataFrame:
    identity = manifest["identity"]
    return pd.DataFrame(
        {
            "id": [str(value) for value in identity["ids"]],
            "plume_id": [str(value) for value in identity["plume_ids"]],
            "event_id": [str(value) for value in identity["event_ids"]],
            "label": [int(value) for value in identity["labels"]],
        }
    )


def _read_prediction(
    path: Path,
    expected: pd.DataFrame,
    *,
    name: str,
    prefer_saved_logit: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    tempo.assert_development_path(path, purpose=f"{name} predictions")
    frame = pd.read_csv(path, low_memory=False)
    required = ("id", "plume_id", "event_id", "label", "probability")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    observed = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "plume_id": frame["plume_id"].astype(str),
            "event_id": frame["event_id"].astype(str),
            "label": frame["label"].astype(int),
        }
    )
    for column in ("id", "plume_id", "event_id", "label"):
        if not observed[column].equals(expected[column]):
            raise ValueError(f"{name}: ordered {column} differs from manifest")
    probability = frame["probability"].to_numpy(dtype=np.float64)
    if (
        probability.shape != (len(expected),)
        or not np.isfinite(probability).all()
        or np.any(probability <= 0)
        or np.any(probability >= 1)
    ):
        raise ValueError(f"{name}: invalid probability vector")
    if prefer_saved_logit and "logit" in frame:
        logit = frame["logit"].to_numpy(dtype=np.float64)
        source = "saved_float32_logit_csv"
    else:
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        logit = np.log(clipped) - np.log1p(-clipped)
        source = "float64_probability_inverse_sigmoid"
    if logit.shape != probability.shape or not np.isfinite(logit).all():
        raise ValueError(f"{name}: invalid logit vector")
    return logit, {
        "path": str(path),
        "sha256": tempo.sha256_file(path),
        "logit_source": source,
    }


def _metric_bundle(
    labels: np.ndarray,
    event_ids: Sequence[str],
    logits: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    probability = 1.0 / (1.0 + np.exp(-clipped))
    return (
        global_tempo.metric_bundle(labels, probability, event_ids),
        probability,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--p5-overlay", required=True)
    parser.add_argument("--d1-predictions", nargs=3, required=True)
    parser.add_argument("--patch-predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    if len(args.patch_predictions) < 2:
        raise ValueError("at least two patch seeds are required")

    manifest_path = Path(args.val_manifest).expanduser().resolve()
    p5_path = Path(args.p5_overlay).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (manifest_path, "validation manifest"),
        (p5_path, "P5 overlay"),
        (output_dir, "audit output"),
    ):
        tempo.assert_development_path(path, purpose=purpose)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = tempo.load_manifest(manifest_path, expected_split="val")
    expected = _identity_frame(manifest)
    labels = expected["label"].to_numpy(dtype=np.int64)
    event_ids = expected["event_id"].tolist()
    p5_tensor, p5_audit = tempo.load_base_overlay(
        p5_path, manifest, expected_split="val"
    )
    p5_logit = p5_tensor.numpy().astype(np.float64)

    d1_logits: list[np.ndarray] = []
    d1_sources: list[dict[str, Any]] = []
    for index, raw_path in enumerate(args.d1_predictions):
        value, source = _read_prediction(
            Path(raw_path).expanduser().resolve(),
            expected,
            name=f"D1 seed {index}",
            prefer_saved_logit=False,
        )
        d1_logits.append(value)
        d1_sources.append(source)

    patch_logits: list[np.ndarray] = []
    patch_sources: list[dict[str, Any]] = []
    for index, raw_path in enumerate(args.patch_predictions):
        value, source = _read_prediction(
            Path(raw_path).expanduser().resolve(),
            expected,
            name=f"patch seed {index}",
            prefer_saved_logit=True,
        )
        patch_logits.append(value)
        patch_sources.append(source)

    d1_ensemble = np.mean(np.stack(d1_logits, axis=0), axis=0)
    patch_ensemble = np.mean(np.stack(patch_logits, axis=0), axis=0)
    patch_ensemble_name = (
        f"patch_{len(patch_logits)}_seed_equal_logit"
    )
    combinations = {
        "p5": p5_logit,
        "d1_three_seed_equal_logit": d1_ensemble,
        patch_ensemble_name: patch_ensemble,
        "p5_d1_fixed_equal_logit": (p5_logit + d1_ensemble) / 2.0,
        "p5_d1_patch_fixed_equal_three_logit": (
            p5_logit + d1_ensemble + patch_ensemble
        )
        / 3.0,
        "p5_d1_patch_two_scale_hierarchical_logit": (
            0.25 * p5_logit
            + 0.25 * d1_ensemble
            + 0.50 * patch_ensemble
        ),
    }
    metrics: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, logits in combinations.items():
        metrics[name], probabilities[name] = _metric_bundle(
            labels, event_ids, logits
        )
    for index, logits in enumerate(d1_logits):
        name = f"d1_seed_{index}"
        metrics[name], probabilities[name] = _metric_bundle(
            labels, event_ids, logits
        )
    for index, logits in enumerate(patch_logits):
        name = f"patch_seed_{index}"
        metrics[name], probabilities[name] = _metric_bundle(
            labels, event_ids, logits
        )

    bootstrap_arms = {
        name: probabilities[name]
        for name in (
            "p5",
            patch_ensemble_name,
            "p5_d1_fixed_equal_logit",
            "p5_d1_patch_fixed_equal_three_logit",
            "p5_d1_patch_two_scale_hierarchical_logit",
        )
    }
    bootstrap_metrics = {
        name: metrics[name] for name in bootstrap_arms
    }
    bootstrap = global_tempo.paired_event_bootstrap_deltas(
        labels,
        event_ids,
        bootstrap_arms,
        bootstrap_metrics,
        (
            (
                "equal_three_minus_p5_d1",
                "p5_d1_patch_fixed_equal_three_logit",
                "p5_d1_fixed_equal_logit",
            ),
            (
                "equal_three_minus_p5",
                "p5_d1_patch_fixed_equal_three_logit",
                "p5",
            ),
            (
                "two_scale_hierarchical_minus_p5_d1",
                "p5_d1_patch_two_scale_hierarchical_logit",
                "p5_d1_fixed_equal_logit",
            ),
            (
                "two_scale_hierarchical_minus_patch",
                "p5_d1_patch_two_scale_hierarchical_logit",
                patch_ensemble_name,
            ),
        ),
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
    )

    prediction_frame = expected.copy()
    for name, logits in combinations.items():
        prediction_frame[f"{name}_logit"] = logits.astype(np.float32)
        prediction_frame[f"{name}_probability"] = probabilities[name]
    prediction_path = output_dir / "fixed_multiseed_predictions.csv"
    l89.atomic_csv_write(prediction_path, prediction_frame)

    audit = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "rows": len(expected),
        "events": len(set(event_ids)),
        "identity_contract": {
            "ordered_id_exact": True,
            "ordered_plume_id_exact": True,
            "ordered_event_id_exact": True,
            "ordered_label_exact": True,
            "patch_manifest_identity_sha256": manifest["identity"][
                "identity_sha256"
            ],
        },
        "fixed_combinations": {
            "d1_three_seed_equal_logit": {
                "seeds": 3,
                "within_family_weights": [1 / 3] * 3,
                "searched": False,
            },
            patch_ensemble_name: {
                "seeds": len(patch_logits),
                "within_family_weights": [
                    1 / len(patch_logits)
                ]
                * len(patch_logits),
                "searched": False,
            },
            "p5_d1_fixed_equal_logit": {
                "p5": 0.5,
                "d1_three_seed_equal_logit": 0.5,
                "searched": False,
            },
            "p5_d1_patch_fixed_equal_three_logit": {
                "p5": 1 / 3,
                "d1_three_seed_equal_logit": 1 / 3,
                "patch_four_seed_equal_logit": 1 / 3,
                "searched": False,
            },
            "p5_d1_patch_two_scale_hierarchical_logit": {
                "p5": 0.25,
                "d1_three_seed_equal_logit": 0.25,
                patch_ensemble_name: 0.50,
                "searched": False,
                "interpretation": (
                    "equal logit weight for the global scale "
                    "(P5+D1)/2 and the patch-local scale"
                ),
            },
        },
        "metrics": metrics,
        "paired_canonical_event_bootstrap": {
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.bootstrap_seed),
            "thresholds_refit_per_replicate": False,
            "development_selected_thresholds_fixed": True,
            "post_selection_diagnostic": True,
            "deltas": bootstrap,
        },
        "sources": {
            "val_manifest": str(manifest_path),
            "val_manifest_sha256": tempo.sha256_file(manifest_path),
            "p5_overlay": str(p5_path),
            "p5_overlay_sha256": tempo.sha256_file(p5_path),
            "p5_overlay_audit": p5_audit,
            "d1_predictions": d1_sources,
            "patch_predictions": patch_sources,
        },
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": tempo.sha256_file(prediction_path),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": tempo.sha256_file(Path(__file__).resolve()),
        "selection_or_weight_search": False,
        "test_or_sealed_read": False,
    }
    output_path = output_dir / "fixed_multiseed_audit.json"
    tempo.atomic_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
