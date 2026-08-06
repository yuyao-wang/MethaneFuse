#!/usr/bin/env python3
"""Leakage-free fixed three-expert validation audit for L89 TEMPO.

This audit does not fit or search any weight.  It reports exactly two
predeclared logit combinations:

* equal experts: 1/3 P5 + 1/3 D1 + 1/3 patch-A1;
* hierarchical: 0.5 P5 + 0.25 D1 + 0.25 patch-A1.

Rows are accepted only when ID, plume ID, canonical event ID and label are
identical to the validated projected-patch manifest in the same order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

import tempo_l89_patch as tempo
from research.pretraining_20260727 import l89_ragged_cls_experiment as l89
from research.pretraining_20260727 import (
    l89_patch_local_rctp_followup as patch_followup,
)


FORMAT_VERSION = "tempo-l89-fixed-three-expert-audit-v1"


def _identity_frame(manifest: dict[str, Any]) -> pd.DataFrame:
    identity = manifest["identity"]
    return pd.DataFrame(
        {
            "id": [str(value) for value in identity["ids"]],
            "plume_id": [str(value) for value in identity["plume_ids"]],
            "event_id": [str(value) for value in identity["event_ids"]],
            "label": [int(value) for value in identity["labels"]],
        }
    )


def _read_and_verify(
    path: Path,
    expected: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    tempo.assert_development_path(path, purpose=f"{name} predictions")
    frame = pd.read_csv(path, low_memory=False)
    required = ("id", "plume_id", "event_id", "label", "probability")
    missing = [column for column in required if column not in frame.columns]
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
    return frame


def _probability_to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _metrics(
    labels: torch.Tensor,
    event_ids: Sequence[str],
    logits: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    metrics, probability = patch_followup.metrics_from_logits(
        labels, tensor, event_ids
    )
    tempo.add_independently_selected_macro_f1(
        metrics, labels, probability, event_ids
    )
    threshold = float(
        metrics["event_balanced_at_event_selected"]["threshold"]
    )
    metrics["all_negative_event_fp"] = tempo._all_negative_fp_audit(
        labels, event_ids, probability, threshold=threshold
    )
    return metrics, probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--p5-overlay", required=True)
    parser.add_argument("--d1-predictions", required=True)
    parser.add_argument("--patch-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.val_manifest).expanduser().resolve()
    p5_path = Path(args.p5_overlay).expanduser().resolve()
    d1_path = Path(args.d1_predictions).expanduser().resolve()
    patch_path = Path(args.patch_predictions).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (manifest_path, "validation manifest"),
        (p5_path, "P5 overlay"),
        (d1_path, "D1 predictions"),
        (patch_path, "patch predictions"),
        (output_dir, "audit output"),
    ):
        tempo.assert_development_path(path, purpose=purpose)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = tempo.load_manifest(manifest_path, expected_split="val")
    expected = _identity_frame(manifest)
    d1 = _read_and_verify(d1_path, expected, name="D1")
    patch = _read_and_verify(patch_path, expected, name="patch-A1")
    p5_logit, p5_audit = tempo.load_base_overlay(
        p5_path, manifest, expected_split="val"
    )
    p5 = p5_logit.numpy().astype(np.float64)
    d1_logit = _probability_to_logit(
        d1["probability"].to_numpy(dtype=np.float64)
    )
    if "logit" in patch.columns:
        patch_logit = patch["logit"].to_numpy(dtype=np.float64)
        patch_logit_source = "saved_float32_logit_csv"
    else:
        patch_logit = _probability_to_logit(
            patch["probability"].to_numpy(dtype=np.float64)
        )
        patch_logit_source = "probability_inverse_sigmoid"
    for name, value in (
        ("P5", p5),
        ("D1", d1_logit),
        ("patch-A1", patch_logit),
    ):
        if value.shape != (len(expected),) or not np.isfinite(value).all():
            raise ValueError(f"{name}: invalid logit vector")

    combinations = {
        "p5": p5,
        "d1": d1_logit,
        "patch_a1": patch_logit,
        "equal_three_expert": (p5 + d1_logit + patch_logit) / 3.0,
        "hierarchical_predeclared": (
            0.50 * p5 + 0.25 * d1_logit + 0.25 * patch_logit
        ),
    }
    labels = torch.tensor(expected["label"].tolist(), dtype=torch.long)
    event_ids = expected["event_id"].tolist()
    all_metrics: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, logits in combinations.items():
        all_metrics[name], probabilities[name] = _metrics(
            labels, event_ids, logits
        )

    prediction_output = expected.copy()
    prediction_output["p5_logit"] = p5.astype(np.float32)
    prediction_output["d1_logit"] = d1_logit.astype(np.float32)
    prediction_output["patch_a1_logit"] = patch_logit.astype(np.float32)
    prediction_output["equal_three_expert_probability"] = probabilities[
        "equal_three_expert"
    ]
    prediction_output["hierarchical_predeclared_probability"] = probabilities[
        "hierarchical_predeclared"
    ]
    prediction_path = output_dir / "fixed_three_expert_predictions.csv"
    l89.atomic_csv_write(prediction_path, prediction_output)

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
            "equal_three_expert": {
                "p5": 1 / 3,
                "d1": 1 / 3,
                "patch_a1": 1 / 3,
                "searched": False,
            },
            "hierarchical_predeclared": {
                "p5": 0.50,
                "d1": 0.25,
                "patch_a1": 0.25,
                "searched": False,
                "interpretation": (
                    "appearance anchor receives half the logit mass; two "
                    "temporal experts split the remaining half"
                ),
            },
        },
        "metrics": all_metrics,
        "sources": {
            "val_manifest": str(manifest_path),
            "val_manifest_sha256": tempo.sha256_file(manifest_path),
            "p5_overlay": str(p5_path),
            "p5_overlay_sha256": tempo.sha256_file(p5_path),
            "p5_overlay_audit": p5_audit,
            "d1_predictions": str(d1_path),
            "d1_predictions_sha256": tempo.sha256_file(d1_path),
            "d1_logit_conversion": (
                "float64 log(p)-log1p(-p), probabilities clipped to [1e-7,1-1e-7]"
            ),
            "patch_predictions": str(patch_path),
            "patch_predictions_sha256": tempo.sha256_file(patch_path),
            "patch_logit_source": patch_logit_source,
        },
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": tempo.sha256_file(prediction_path),
        "selection_or_weight_search": False,
        "test_or_sealed_read": False,
    }
    output_path = output_dir / "fixed_three_expert_audit.json"
    tempo.atomic_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
