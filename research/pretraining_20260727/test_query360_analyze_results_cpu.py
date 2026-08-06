#!/usr/bin/env python3
"""CPU-only tests for Query360 canonical-event paired analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import query360_analyze_results as analysis


def _prediction_frame(
    *,
    arm: str,
    condition: str,
    epoch: int,
    probabilities: np.ndarray,
    frozen_seed: int | None,
) -> pd.DataFrame:
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "id": [str(index) for index in range(len(labels))],
            "plume_id": [
                f"acquisition-{index // 2}-A" for index in range(len(labels))
            ],
            "cluster_id": [f"cluster-{index // 2}" for index in range(len(labels))],
            "macro_region_id": ["region-1"] * len(labels),
            "availability_signature": ["s2+l89"] * len(labels),
            "label": labels,
            "probability": probabilities,
            "prediction_at_0_5": (probabilities >= 0.5).astype(np.int64),
            "arm": [arm] * len(labels),
            "epoch": [epoch] * len(labels),
        }
    )
    if frozen_seed is not None:
        frame["seed"] = frozen_seed
    else:
        frame["condition"] = condition
    return frame


def _validation(frame: pd.DataFrame) -> dict[str, object]:
    metrics = analysis.binary_metrics(frame["label"], frame["probability"])
    return {
        "overall": {
            **metrics,
            "rows": len(frame),
        }
    }


class Query360AnalyzeResultsTests(unittest.TestCase):
    def test_01_metrics_match_sklearn_with_ties(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.8, 0.7, 0.8, 0.4, 0.4, 0.7, 0.2])
        actual = analysis.binary_metrics(labels, probabilities)
        self.assertAlmostEqual(
            actual["ap"], average_precision_score(labels, probabilities), places=12
        )
        self.assertAlmostEqual(
            actual["auc"], roc_auc_score(labels, probabilities), places=12
        )
        self.assertAlmostEqual(
            actual["macro_f1_at_0_5"],
            f1_score(
                labels,
                probabilities >= 0.5,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            ),
            places=12,
        )

    def test_02_canonical_event_and_pairing_fail_closed(self) -> None:
        self.assertEqual(
            analysis.canonical_event_id("GAO20240101t000000p0000-AB"),
            "GAO20240101t000000p0000",
        )
        base = _prediction_frame(
            arm="current_only",
            condition="panopticon_pretrained",
            epoch=1,
            probabilities=np.linspace(0.1, 0.9, 12),
            frozen_seed=17,
        )
        base["event_id"] = base["plume_id"].map(analysis.canonical_event_id)
        candidate = base.copy()
        candidate.loc[0, "label"] = 1
        with self.assertRaisesRegex(analysis.AnalysisError, "paired label"):
            analysis.validate_paired_frames(
                base, candidate, context="synthetic mismatch"
            )
        candidate = base.copy()
        candidate.loc[0, "event_id"] = "wrong-event"
        with self.assertRaisesRegex(analysis.AnalysisError, "paired event_id"):
            analysis.validate_paired_frames(
                base, candidate, context="synthetic mismatch"
            )

    def _write_frozen_summary(
        self,
        root: Path,
        *,
        condition: str,
        seed_probabilities: dict[
            int, tuple[np.ndarray, np.ndarray]
        ],
    ) -> Path:
        output = root / f"frozen_{condition}"
        seeds = list(seed_probabilities)
        seed_results = []
        for seed, (current_probabilities, tq_probabilities) in (
            seed_probabilities.items()
        ):
            arm_results = {}
            for arm, probabilities in (
                ("current_only", current_probabilities),
                ("transient_query", tq_probabilities),
            ):
                frame = _prediction_frame(
                    arm=arm,
                    condition=condition,
                    epoch=1,
                    probabilities=probabilities,
                    frozen_seed=seed,
                )
                path = (
                    output
                    / f"seed_{seed}"
                    / arm
                    / "validation_best_ap_predictions.csv"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                arm_results[arm] = {
                    "epoch": 1,
                    "validation": _validation(frame),
                }
            seed_results.append({"seed": seed, "arms": arm_results})
        dataset_contract = {
            "train": {
                "manifest_sha256": "a" * 64,
                "rows": 12,
            },
            "inner_val": {
                "manifest_sha256": "b" * 64,
                "rows": 12,
            },
            "sensor_names": ["s2", "l89", "emit", "s5p"],
            "role_names": ["current", "short_history", "long_history"],
        }
        training_contract = {
            "arms": ["current_only", "transient_query"],
            "seeds": seeds,
            "epochs": 1,
        }
        summary = {
            "schema_version": "query360-head-summary-v1",
            "encoder": {"condition": condition},
            "arms": ["current_only", "transient_query"],
            "seeds": seeds,
            "dataset_contract": dataset_contract,
            "training_contract": training_contract,
            "seed_results": seed_results,
            "safety": {"external_test_manifest_read": False},
        }
        path = output / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def _write_online_summary(
        self,
        root: Path,
        *,
        condition: str,
        weak: np.ndarray,
        strong: np.ndarray,
        backbone_lr: float,
    ) -> Path:
        output = root / f"online_{condition}"
        arm_results = {}
        for arm, probabilities in (
            ("current_only", weak),
            ("transient_query", strong),
        ):
            frame = _prediction_frame(
                arm=arm,
                condition=condition,
                epoch=1,
                probabilities=probabilities,
                frozen_seed=None,
            )
            path = output / arm / "validation_best_ap_predictions.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            arm_results[arm] = {
                "epoch": 1,
                "validation": _validation(frame),
            }
        summary = {
            "schema_version": "query360-online-summary-v1",
            "condition": condition,
            "full_backbone_train": True,
            "arms": arm_results,
            "train_manifest": {
                "path": "/safe/inner_train.csv",
                "sha256": "a" * 64,
                "rows": 12,
            },
            "inner_val_manifest": {
                "path": "/safe/inner_val.csv",
                "sha256": "b" * 64,
                "rows": 12,
            },
            "training_contract": {
                "requested_arms": ["current_only", "transient_query"],
                "epochs": 1,
                "max_eval_steps": 0,
                "batch_seed": 17,
                "backbone_lr": backbone_lr,
                "head_lr": 3e-4,
            },
            "safety": {"external_test_manifest_read": False},
        }
        path = output / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_03_end_to_end_writes_deterministic_paired_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_analysis_") as temporary:
            root = Path(temporary)
            labels = np.asarray([0, 1] * 6)
            strong = np.where(labels == 1, 0.9, 0.1).astype(np.float64)
            weak = np.asarray(
                [0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.75, 0.25, 0.4, 0.6, 0.6, 0.4],
                dtype=np.float64,
            )
            medium = np.asarray(
                [0.2, 0.8, 0.7, 0.3, 0.3, 0.7, 0.65, 0.35, 0.4, 0.6, 0.55, 0.45],
                dtype=np.float64,
            )
            inverse = 1.0 - strong
            panopticon_seed_probabilities = {
                17: (weak, strong),
                29: (medium, strong),
                43: (inverse, medium),
            }
            random_seed_probabilities = {
                17: (weak, medium),
                29: (medium, weak),
                43: (inverse, weak),
            }
            frozen = [
                self._write_frozen_summary(
                    root,
                    condition="panopticon_pretrained",
                    seed_probabilities=panopticon_seed_probabilities,
                ),
                self._write_frozen_summary(
                    root,
                    condition="random_frozen",
                    seed_probabilities=random_seed_probabilities,
                ),
            ]
            online = [
                self._write_online_summary(
                    root,
                    condition="panopticon_pretrained",
                    weak=medium,
                    strong=strong,
                    backbone_lr=1e-5,
                ),
                self._write_online_summary(
                    root,
                    condition="scratch",
                    weak=weak,
                    strong=medium,
                    backbone_lr=1e-4,
                ),
            ]
            first = analysis.analyze(
                frozen_summaries=frozen,
                online_summaries=online,
                output_dir=root / "analysis_a",
                repeats=analysis.MIN_BOOTSTRAP_REPEATS,
                seed=123,
            )
            second = analysis.analyze(
                frozen_summaries=frozen,
                online_summaries=online,
                output_dir=root / "analysis_b",
                repeats=analysis.MIN_BOOTSTRAP_REPEATS,
                seed=123,
            )
            first_payload = json.loads(first["json"].read_text(encoding="utf-8"))
            second_payload = json.loads(second["json"].read_text(encoding="utf-8"))
            self.assertEqual(
                first_payload["contrasts"], second_payload["contrasts"]
            )
            self.assertEqual(first_payload["bootstrap"]["accepted_repeats"], 2_000)
            self.assertEqual(first_payload["bootstrap"]["canonical_events"], 6)
            self.assertTrue(first["bootstrap_csv"].is_file())
            self.assertTrue(first["artifact_csv"].is_file())
            rows = pd.read_csv(first["bootstrap_csv"])
            target = rows[
                (rows["kind"] == "within_condition_arm_effect")
                & (rows["metric"] == "auc")
                & rows["contrast_id"].str.contains(
                    "panopticon_pretrained\\|transient_query", regex=True
                )
            ]
            self.assertFalse(target.empty)
            self.assertTrue((target["point_delta"] > 0).all())

            aggregate = rows[
                (rows["family"] == "frozen")
                & (
                    rows["kind"]
                    == "aggregate_within_condition_arm_effect"
                )
                & (rows["metric"] == "ap")
                & rows["contrast_id"].str.contains(
                    "panopticon_pretrained\\|transient_query-minus-current_only",
                    regex=True,
                )
            ]
            per_seed = rows[
                (rows["family"] == "frozen")
                & (rows["kind"] == "within_condition_arm_effect")
                & (rows["metric"] == "ap")
                & rows["contrast_id"].str.contains(
                    "panopticon_pretrained\\|transient_query-minus-current_only",
                    regex=True,
                )
            ]
            self.assertEqual(len(per_seed), 3)
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(int(aggregate.iloc[0]["seed"]), -1)
            self.assertEqual(
                aggregate.iloc[0]["aggregate_method"],
                "equal_mean_of_seed_metric_deltas",
            )
            self.assertEqual(int(aggregate.iloc[0]["aggregate_seed_count"]), 3)
            self.assertAlmostEqual(
                float(aggregate.iloc[0]["point_delta"]),
                float(per_seed["point_delta"].mean()),
                places=12,
            )

            # A nonlinear metric applied after probability ensembling is a
            # different estimand and must not accidentally define aggregate.
            mean_current_probability = np.mean(
                np.stack(
                    [
                        pair[0]
                        for pair in panopticon_seed_probabilities.values()
                    ]
                ),
                axis=0,
            )
            mean_tq_probability = np.mean(
                np.stack(
                    [
                        pair[1]
                        for pair in panopticon_seed_probabilities.values()
                    ]
                ),
                axis=0,
            )
            probability_ensemble_delta = (
                analysis.binary_metrics(labels, mean_tq_probability)["ap"]
                - analysis.binary_metrics(labels, mean_current_probability)["ap"]
            )
            self.assertNotAlmostEqual(
                float(aggregate.iloc[0]["point_delta"]),
                probability_ensemble_delta,
                places=6,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
