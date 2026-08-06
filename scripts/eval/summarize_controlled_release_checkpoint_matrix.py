#!/usr/bin/env python3
"""Audit and summarize the controlled-release S2 checkpoint matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_RULE = "original two-class logit argmax (no fitted threshold stored)"

DATASETS = {
    "original_table_57": {
        "manifest": REPO_ROOT
        / "data/controlled_release_test/combined_original_labels_legacy480_exact/s2_model_manifest.csv",
        "rows": 57,
        "positives": 12,
        "negatives": 45,
        "files": {
            "Stage1 shared": "controlled_release_s2_original_labels_legacy480_exact_stage1_predictions.csv",
            "Universal 480m": "controlled_release_s2_original_labels_legacy480_exact_universal_predictions.csv",
            "S2-only 480m": "controlled_release_s2_original_labels_legacy480_exact_s2_only_predictions.csv",
            "Q/V-LoRA-MoE": "controlled_release_s2_original_labels_legacy480_exact_qv_loramoe_predictions.csv",
            "Q/V-LoRA": "controlled_release_s2_original_labels_legacy480_exact_qv_lora_predictions.csv",
            "Post-block MoE": "controlled_release_s2_original_labels_legacy480_exact_postblock_moe_predictions.csv",
        },
    },
    "distance_crops_384": {
        "manifest": REPO_ROOT
        / "data/controlled_release_test/label1_legacy_balanced_480m/manifest.csv",
        "rows": 384,
        "positives": 192,
        "negatives": 192,
        "files": {
            "Stage1 shared": "controlled_release_s2_label1_scene_balanced_stage1_480m_predictions.csv",
            "Universal 480m": "controlled_release_s2_label1_scene_balanced_480m_predictions.csv",
            "S2-only 480m": "controlled_release_s2_label1_scene_balanced_s2_only_480m_predictions.csv",
            "Q/V-LoRA-MoE": "controlled_release_s2_label1_scene_balanced_qv_loramoe_480m_predictions.csv",
            "Q/V-LoRA": "controlled_release_s2_label1_scene_balanced_qv_lora_480m_predictions.csv",
            "Post-block MoE": "controlled_release_s2_label1_scene_balanced_postblock_moe_480m_predictions.csv",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results/eval"
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def prediction_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if {"prediction", "positive_probability"} <= set(frame.columns):
        return "prediction", "positive_probability"
    if {"s2_only_prediction", "s2_only_probability"} <= set(frame.columns):
        return "s2_only_prediction", "s2_only_probability"
    raise ValueError(f"Unrecognized prediction columns: {frame.columns.tolist()}")


def summarize(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    return {
        "samples": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(tp / positives),
        "specificity": float(tn / negatives),
        "fpr": float(fp / negatives),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "positive_probability_mean": float(scores[labels == 1].mean()),
        "negative_probability_mean": float(scores[labels == 0].mean()),
    }


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    audit: dict[str, dict] = {}

    for dataset_name, spec in DATASETS.items():
        manifest = pd.read_csv(spec["manifest"], low_memory=False)
        expected_labels = manifest["label"].astype(np.int64).to_numpy()
        if len(manifest) != spec["rows"]:
            raise ValueError(f"{dataset_name}: unexpected manifest row count")
        if int((expected_labels == 1).sum()) != spec["positives"]:
            raise ValueError(f"{dataset_name}: unexpected positive count")
        if int((expected_labels == 0).sum()) != spec["negatives"]:
            raise ValueError(f"{dataset_name}: unexpected negative count")

        audit[dataset_name] = {
            "manifest": str(spec["manifest"]),
            "rows": int(len(manifest)),
            "positives": int((expected_labels == 1).sum()),
            "negatives": int((expected_labels == 0).sum()),
            "models_checked": [],
        }
        expected_ids = manifest["id"].astype(str).tolist()

        for model_name, filename in spec["files"].items():
            path = args.results_dir / filename
            frame = pd.read_csv(path, low_memory=False)
            pred_col, score_col = prediction_columns(frame)
            labels = frame["label"].astype(np.int64).to_numpy()
            preds = frame[pred_col].astype(np.int64).to_numpy()
            scores = frame[score_col].astype(np.float64).to_numpy()

            if len(frame) != len(manifest):
                raise ValueError(f"{dataset_name}/{model_name}: row count mismatch")
            if frame["id"].astype(str).tolist() != expected_ids:
                raise ValueError(f"{dataset_name}/{model_name}: row ID/order mismatch")
            if not np.array_equal(labels, expected_labels):
                raise ValueError(f"{dataset_name}/{model_name}: label mismatch")
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError(f"{dataset_name}/{model_name}: invalid probabilities")

            record = {
                "dataset": dataset_name,
                "model": model_name,
                "decision_rule": DECISION_RULE,
                "prediction_file": str(path.resolve()),
                **summarize(labels, preds, scores),
            }
            records.append(record)
            audit[dataset_name]["models_checked"].append(model_name)

    output = pd.DataFrame.from_records(records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "audit": audit,
                "decision_rule": {
                    "kind": "two_class_logit_argmax",
                    "source": "original training/evaluation code",
                    "fitted_threshold_present": False,
                    "test_set_threshold_tuning": False,
                    "probability_equivalent_threshold": 0.5,
                },
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
