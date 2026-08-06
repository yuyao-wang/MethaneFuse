#!/usr/bin/env python3
"""Create one SHA-backed table for completed 64/128-D TEMPO patch heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import tempo_l89_patch as tempo
from research.pretraining_20260727 import (
    l89_patch_local_rctp_followup as patch_followup,
)


FORMAT_VERSION = "tempo-l89-patch-head-summary-v1"


def _metrics(
    labels: torch.Tensor,
    event_ids: list[str],
    probability: np.ndarray,
) -> dict[str, Any]:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7
    )
    logits = torch.from_numpy(
        (np.log(clipped) - np.log1p(-clipped)).astype(np.float32)
    )
    metrics, reproduced = patch_followup.metrics_from_logits(
        labels, logits, event_ids
    )
    tempo.add_independently_selected_macro_f1(
        metrics, labels, reproduced, event_ids
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    tempo.assert_development_path(run_root, purpose="patch run root")
    tempo.assert_development_path(output, purpose="patch summary")
    manifest_path = run_root / "cache" / "val" / "manifest.json"
    manifest = tempo.load_manifest(manifest_path, expected_split="val")
    expected_ids = [str(value) for value in manifest["identity"]["ids"]]
    expected_events = [
        str(value) for value in manifest["identity"]["event_ids"]
    ]
    expected_labels = [
        int(value) for value in manifest["identity"]["labels"]
    ]
    labels = torch.tensor(expected_labels, dtype=torch.long)

    rows: list[dict[str, Any]] = []
    for result_path in sorted((run_root / "heads").glob("*/result.json")):
        arm_dir = result_path.parent
        prediction_path = arm_dir / "validation_predictions.csv"
        if not prediction_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        prediction = pd.read_csv(prediction_path, low_memory=False)
        if prediction["id"].astype(str).tolist() != expected_ids:
            raise ValueError(f"{arm_dir.name}: ordered IDs differ")
        if prediction["event_id"].astype(str).tolist() != expected_events:
            raise ValueError(f"{arm_dir.name}: ordered events differ")
        if prediction["label"].astype(int).tolist() != expected_labels:
            raise ValueError(f"{arm_dir.name}: labels differ")
        primary = _metrics(
            labels,
            expected_events,
            prediction["probability"].to_numpy(dtype=np.float64),
        )
        fusion_column = "fixed_0p5_late_logit_fusion_probability"
        fusion = (
            _metrics(
                labels,
                expected_events,
                prediction[fusion_column].to_numpy(dtype=np.float64),
            )
            if fusion_column in prediction.columns
            else None
        )
        configuration = result["configuration"]
        row = {
            "arm": arm_dir.name,
            "selected_epoch": int(prediction["selected_epoch"].iloc[0]),
            "projection_dim": int(
                manifest["configuration"]["projection"]["output_dim"]
            ),
            "radius": int(configuration["radius"]),
            "neighbourhood": (
                f"{2 * int(configuration['radius']) + 1}x"
                f"{2 * int(configuration['radius']) + 1}"
            ),
            "topk_fraction": float(configuration["topk_fraction"]),
            "normality_features": bool(
                configuration["use_normality_features"]
            ),
            "null_weight": float(configuration["null_weight"]),
            "base_family": str(configuration["base_overlay_family"]),
            "event_balanced_ap": float(primary["event_balanced_ap"]),
            "event_balanced_auc": float(primary["event_balanced_auc"]),
            "event_balanced_macro_f1_selected": float(
                primary["event_balanced_macro_f1_selected"]
            ),
            "event_balanced_macro_f1_threshold": float(
                primary["event_balanced_macro_f1_selected_threshold"]
            ),
            "event_balanced_positive_f1_selected": float(
                primary["event_balanced_positive_f1_selected"]
            ),
            "fusion_event_balanced_ap": (
                float(fusion["event_balanced_ap"])
                if fusion is not None
                else None
            ),
            "fusion_event_balanced_macro_f1_selected": (
                float(fusion["event_balanced_macro_f1_selected"])
                if fusion is not None
                else None
            ),
            "result": str(result_path),
            "result_sha256": tempo.sha256_file(result_path),
            "predictions": str(prediction_path),
            "predictions_sha256": tempo.sha256_file(prediction_path),
        }
        rows.append(row)
    if not rows:
        raise RuntimeError("no complete heads found")
    table = pd.DataFrame(rows).sort_values(
        ["event_balanced_ap", "arm"], ascending=[False, True]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    tempo.atomic_json(
        output,
        {
            "format_version": FORMAT_VERSION,
            "status": "complete",
            "run_root": str(run_root),
            "val_manifest": str(manifest_path),
            "val_manifest_sha256": tempo.sha256_file(manifest_path),
            "projection": manifest["configuration"]["projection"],
            "rows": int(manifest["rows"]),
            "events": len(set(expected_events)),
            "heads": rows,
            "selection": (
                "Each head/epoch selected by development event-balanced AP; "
                "macro-F1 threshold independently maximized with the global "
                "L89 protocol."
            ),
            "test_or_sealed_read": False,
        },
    )
    # JSON is the canonical rich artifact; CSV is its human-readable table.
    from research.pretraining_20260727 import (
        l89_ragged_cls_experiment as l89,
    )

    l89.atomic_csv_write(csv_path, table)
    print(
        json.dumps(
            {
                "json": str(output),
                "json_sha256": tempo.sha256_file(output),
                "csv": str(csv_path),
                "csv_sha256": tempo.sha256_file(csv_path),
                "heads": len(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
