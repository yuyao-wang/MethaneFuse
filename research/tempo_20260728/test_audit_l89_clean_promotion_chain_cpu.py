#!/usr/bin/env python3
"""CPU-only tests for the clean L89 promotion-chain audit."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache
from research.tempo_20260728 import audit_l89_clean_promotion_chain as target


def metric(
    ap: float, auc: float, macro: float, fp_mass: float = 2.0
) -> dict[str, float]:
    return {
        "event_balanced_ap": ap,
        "event_balanced_auc": auc,
        "event_balanced_selected_macro_f1": macro,
        "event_balanced_selected_threshold": 0.5,
        "all_negative_fp_mass": fp_mass,
    }


def fake_bootstrap(
    labels: np.ndarray,
    event_ids: list[str],
    probabilities: dict[str, np.ndarray],
    point_metrics: dict[str, dict[str, float]],
    comparisons: tuple[tuple[str, str, str], ...],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    del labels, event_ids, probabilities
    if replicates != 5000 or seed != 2026072808:
        raise AssertionError("formal bootstrap constants changed")
    names = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_selected",
        "event_balanced_macro_f1_selected",
        "all_negative_fp_mass",
    )
    output: dict[str, object] = {}
    for comparison, left, right in comparisons:
        metrics: dict[str, object] = {}
        for name in names:
            point = float(point_metrics[left][name] - point_metrics[right][name])
            metrics[name] = {
                "point": point,
                "ci_95_low": point - 0.01,
                "ci_95_high": point + 0.01,
                "win_probability": 0.75,
                "better_direction": (
                    "lower" if name == "all_negative_fp_mass" else "higher"
                ),
            }
        output[comparison] = metrics
    return output


class PromotionChainTests(unittest.TestCase):
    def test_fixed_fallback_uses_mean_d1_logits_before_fusion(self) -> None:
        p0 = np.asarray([0.8, 0.2], dtype=np.float64)
        d1 = [
            np.asarray([0.9, 0.4]),
            np.asarray([0.6, 0.3]),
            np.asarray([0.7, 0.8]),
        ]
        mean_d1, candidate = target.fixed_reference_mean_d1(p0, d1)
        expected_mean = np.mean(
            np.stack([target.fixed.logit(value) for value in d1]), axis=0
        )
        np.testing.assert_allclose(
            target.fixed.logit(mean_d1), expected_mean, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            candidate,
            target.fixed.sigmoid(
                0.5 * target.fixed.logit(p0) + 0.5 * expected_mean
            ),
            rtol=0,
            atol=1e-12,
        )

    def test_p5_qualification_requires_every_condition(self) -> None:
        head = {
            "p0": metric(0.70, 0.70, 0.72, 2.0),
            "p4": metric(0.71, 0.71, 0.73),
            "p5": metric(0.75, 0.76, 0.728, 2.5),
        }
        passed = target.evaluate_p5_qualification(
            p4_synthetic_ap=0.70,
            p5_synthetic_ap=0.71,
            formal_gate_metrics=head,
        )
        self.assertTrue(passed["P5_qualified"])
        self.assertTrue(
            all(item["pass"] for item in passed["conditions"].values())
        )
        failed = target.evaluate_p5_qualification(
            p4_synthetic_ap=0.70,
            p5_synthetic_ap=0.70,
            formal_gate_metrics=head,
        )
        self.assertFalse(failed["P5_qualified"])
        self.assertFalse(
            failed["conditions"]["synthetic_ap_p5_gt_p4"]["pass"]
        )

    def test_promotion_gate_strict_and_tolerant_boundaries(self) -> None:
        p5 = {
            "event_balanced_ap": 0.70,
            "event_balanced_auc": 0.72,
            "event_balanced_macro_f1_selected": 0.75,
            "all_negative_fp_mass": 2.0,
        }
        candidate = {
            "event_balanced_ap": 0.71,
            "event_balanced_auc": 0.73,
            "event_balanced_macro_f1_selected": 0.747,
            "all_negative_fp_mass": 2.5,
        }
        bootstrap = {
            "candidate_minus_p5": {
                "event_balanced_ap": {"ci_95_low": 0.0001}
            }
        }
        gate = target.evaluate_promotion_gate(
            point_metrics={
                "p5": p5,
                "fixed_p5_mean_d1": candidate,
            },
            bootstrap=bootstrap,
        )
        self.assertTrue(gate["promotion"])
        tied = dict(candidate)
        tied["event_balanced_ap"] = p5["event_balanced_ap"]
        gate = target.evaluate_promotion_gate(
            point_metrics={"p5": p5, "fixed_p5_mean_d1": tied},
            bootstrap=bootstrap,
        )
        self.assertFalse(gate["promotion"])

    def test_outer_like_path_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "outer" / "artifact.json"
            with self.assertRaisesRegex(ValueError, "held-out/outer"):
                target.assert_inner_path(bad, purpose="fixture")

    def test_sidecar_summary_checkpoint_and_initial_state_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "p4.pt"
            checkpoint = {
                "arm": "p4_response_scrambled",
                "epoch": 1,
                "best_dev_metrics": {"average_precision": 0.72},
                "combined_initial_state_sha256": "fresh",
                "sidecar_state": {"weight": torch.zeros(1)},
                "test_or_sealed_or_holdout_read": False,
            }
            cache.atomic_torch_save(checkpoint_path, checkpoint)
            summary_path = root / "p4.json"
            cache.atomic_json_write(
                summary_path,
                {
                    "arm": "p4_response_scrambled",
                    "best_epoch": 1,
                    "best_dev_metrics": {"average_precision": 0.72},
                    "sidecar_checkpoint": str(checkpoint_path),
                    "sidecar_checkpoint_sha256": cache.sha256_file(
                        checkpoint_path
                    ),
                    "combined_initial_state_sha256": "fresh",
                    "test_or_sealed_or_holdout_read": False,
                },
            )
            evidence = target.extract_sidecar_capability(
                arm="p4",
                summary_path=summary_path,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(evidence["synthetic_capability_ap"], 0.72)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["best_dev_metrics"]["average_precision"] = 0.71
            cache.atomic_json_write(summary_path, payload)
            with self.assertRaisesRegex(ValueError, "capability AP mismatch"):
                target.extract_sidecar_capability(
                    arm="p4",
                    summary_path=summary_path,
                    checkpoint_path=checkpoint_path,
                )

    def test_unqualified_run_automatically_writes_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            output = Path(args.output_dir)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), contextlib.redirect_stdout(io.StringIO()):
                result = target.run(args)
            self.assertFalse(result["p5_qualification"]["P5_qualified"])
            self.assertFalse(result["promotion_eligible"])
            self.assertFalse(result["promotion"])
            self.assertTrue(result["fallback_triggered"])
            self.assertEqual(
                result["audit_type"], "P0+D1 mechanism-only fallback"
            )
            self.assertTrue(
                (output / "fixed_p0_mean_d1_predictions.csv").is_file()
            )
            self.assertTrue((output / "RESULT.json").is_file())
            self.assertTrue((output / "RESULT.md").is_file())
            self.assertTrue(
                result["validated_upstream_receipts"][
                    "d1_formal_validation"
                ]["all_seed_patience_1_max4_receipts_valid"]
            )
            receipts = result["validated_upstream_receipts"]
            self.assertTrue(
                receipts["fixed_ensemble_p0_probability_replay"]["pass"]
            )
            self.assertEqual(
                set(receipts["fresh_head_formal_gate_metrics"]),
                {"p0", "p4", "p5"},
            )
            self.assertTrue(
                receipts["d1_formal_validation"][
                    "all_seed_sibling_p0_probability_replays_match_fresh_head"
                ]
            )
            self.assertEqual(
                set(
                    receipts["d1_formal_validation"][
                        "per_seed_sibling_p0_probability_replays"
                    ]
                ),
                {str(seed) for seed in target.FROZEN_SEEDS},
            )

    def test_qualified_main_branch_still_requires_d1_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            forced_qualification = {
                "P5_qualified": True,
                "all_conditions_required": True,
                "conditions": {},
                "source_values": {},
            }
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), mock.patch.object(
                target,
                "evaluate_p5_qualification",
                return_value=forced_qualification,
            ), contextlib.redirect_stdout(io.StringIO()):
                result = target.run(args)
            self.assertEqual(result["audit_type"], "P5+D1 primary promotion audit")
            self.assertFalse(result["fallback_triggered"])
            self.assertTrue(
                result["validated_upstream_receipts"][
                    "d1_formal_validation"
                ]["validated_for_all_promotion_branches"]
            )
            self.assertEqual(
                set(result["d1_mechanism_diagnostics"]),
                {str(seed) for seed in target.FROZEN_SEEDS},
            )

    def test_p4_identity_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            p4_path = Path(args.p4_predictions)
            p4 = pd.read_csv(p4_path)
            p4.loc[0, "plume_id"] = "tampered-plume"
            cache.atomic_csv_write(p4_path, p4)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), self.assertRaisesRegex(ValueError, "ordered plume_id"):
                target.run(args)

    def test_fixed_and_per_seed_p0_probability_tamper_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            replay_path = Path(args.p0_predictions)
            replay = pd.read_csv(replay_path)
            replay.loc[0, "probability"] += 0.01
            cache.atomic_csv_write(replay_path, replay)
            with self.assertRaisesRegex(
                ValueError, "fixed_ensemble_p0 P0 replay probability"
            ):
                target.run(args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            replay_path = (
                root
                / "d1"
                / "seed_20260728"
                / "p0_base_predictions.csv"
            )
            replay = pd.read_csv(replay_path)
            replay.loc[1, "probability"] += 0.02
            cache.atomic_csv_write(replay_path, replay)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), self.assertRaisesRegex(
                ValueError,
                "d1_seed_20260728_sibling_p0 P0 replay probability",
            ):
                target.run(args)

    def test_shuffle_availability_stratum_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            summary_path = (
                root / "d1" / "seed_20260729" / "summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validity = summary["results"]["d1_gated_delta"][
                "history_shuffle_validity"
            ]
            validity["availability_mismatch_rows"] = 1
            validity["availability_pattern_mismatch_count"] = 1
            cache.atomic_json_write(summary_path, summary)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), self.assertRaisesRegex(
                ValueError, "history shuffle identity/label/shape"
            ):
                target.run(args)

    def test_d1_history_sha_and_parent_model_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            summary_path = (
                root / "d1" / "seed_20260727" / "summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["results"]["d1_gated_delta"]["early_stop_receipt"][
                "history_sha256"
            ] = "0" * 64
            cache.atomic_json_write(summary_path, summary)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), self.assertRaisesRegex(ValueError, "history SHA mismatch"):
                target.run(args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            summary_path = (
                root / "d1" / "seed_20260727" / "summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["frozen_role_only_base_audit"][
                "model_state_sha256"
            ] = "f" * 64
            summary["results"]["p0_base"]["frozen_role_only_base_audit"][
                "model_state_sha256"
            ] = "f" * 64
            cache.atomic_json_write(summary_path, summary)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), self.assertRaisesRegex(ValueError, "exact fresh P0 parent"):
                target.run(args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prepare_full_fixture(root)
            summary_path = (
                root / "d1" / "seed_20260727" / "summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for receipt in (
                summary[
                    "zero_initialized_d1_prototype_exact_p0_replay"
                ],
                summary["results"]["p0_base"][
                    "zero_initialized_d1_prototype_exact_p0_replay"
                ],
                summary["results"]["d1_gated_delta"][
                    "epoch_zero_exact_p0_replay"
                ],
            ):
                receipt["maximum_absolute_logit_error"] = 1.0
            cache.atomic_json_write(summary_path, summary)
            with mock.patch.object(
                target.fixed,
                "fixed_threshold_event_bootstrap",
                side_effect=fake_bootstrap,
            ), self.assertRaisesRegex(
                ValueError, "zero-init prototype did not exactly replay P0"
            ):
                target.run(args)

    def _prepare_full_fixture(
        self, root: Path
    ) -> tuple[argparse.Namespace, dict[str, pd.DataFrame]]:
        p4_checkpoint = self._write_sidecar(root, "p4", 0.80)
        p5_checkpoint = self._write_sidecar(root, "p5", 0.70)
        frames = self._write_predictions(root)
        head_metrics = {
            arm: self._head_metric(frame)
            for arm, frame in {
                "p0": frames["p0"],
                "p4": frames["p4"],
                "p5": frames["p5"],
            }.items()
        }
        head_path = root / "comparison.json"
        head_prediction_paths = {
            "p0": root / "p0.csv",
            "p4": root / "p4.csv",
            "p5": root / "p5.csv",
        }
        cache.atomic_json_write(
            head_path,
            {
                "script_version": "l89-clean-inner-matched-head-v1",
                "clean_inner_replicate_exploratory": True,
                "post_hoc_exploratory": False,
                "test_or_sealed_read": False,
                "matching": {
                    "identity_and_labels_equal": True,
                    "same_initial_state": True,
                    "same_parameter_signature": True,
                    "same_batch_plan_sha256": True,
                    "same_seed_optimizer_epochs": True,
                },
                "point_metrics": head_metrics,
                "prediction_provenance": {
                    arm: {
                        "path": str(path),
                        "sha256": cache.sha256_file(path),
                    }
                    for arm, path in head_prediction_paths.items()
                },
            },
        )
        p0_checkpoint = self._write_p0_parent(root, frames["p0"])
        for seed in target.FROZEN_SEEDS:
            self._write_d1_summary(
                root / "d1" / f"seed_{seed}" / "summary.json",
                seed,
                frames["p0"],
                frames[f"d1_seed_{seed}"],
                p0_checkpoint,
                root / "p0.csv",
            )
        cache.atomic_json_write(
            root / "d1" / "run_config.json",
            {
                "seeds": "20260727,20260728,20260729",
                "resolved_arms": ["p0_base", "d1_gated_delta"],
                "base_kind": "event_balanced_p0",
                "event_base_checkpoint": str(p0_checkpoint),
                "epochs": 4,
                "patience": 1,
                "test_or_sealed_or_holdout_read": False,
            },
        )
        cache.atomic_json_write(
            root / "d1" / "run_status.json",
            {
                "status": "complete",
                "seeds": list(target.FROZEN_SEEDS),
                "arms": ["p0_base", "d1_gated_delta"],
                "test_or_sealed_or_holdout_read": False,
            },
        )
        fixed_result_path = root / "fixed_result.json"
        self._write_fixed_result(fixed_result_path, root, frames)
        args = argparse.Namespace(
            p4_sidecar_summary=str(root / "p4_summary.json"),
            p4_sidecar_checkpoint=str(p4_checkpoint),
            p5_sidecar_summary=str(root / "p5_summary.json"),
            p5_sidecar_checkpoint=str(p5_checkpoint),
            head_comparison=str(head_path),
            fixed_ensemble_result=str(fixed_result_path),
            p0_predictions=str(root / "p0_replay.csv"),
            p0_head_predictions=str(root / "p0.csv"),
            p4_predictions=str(root / "p4.csv"),
            p5_predictions=str(root / "p5.csv"),
            d1_template=str(
                root / "d1" / "seed_{seed}" / "d1.csv"
            ),
            d1_summary_template=None,
            output_dir=str(root / "promotion_audit"),
        )
        return args, frames

    @staticmethod
    def _write_sidecar(root: Path, arm: str, ap: float) -> Path:
        artifact_arm = target.SIDECAR_ARTIFACT_ARMS[arm]
        checkpoint_path = root / f"{arm}.pt"
        checkpoint = {
            "arm": artifact_arm,
            "epoch": 1,
            "best_dev_metrics": {"average_precision": ap},
            "combined_initial_state_sha256": "shared-fresh-initial-state",
            "sidecar_state": {"weight": torch.zeros(1)},
            "test_or_sealed_or_holdout_read": False,
        }
        cache.atomic_torch_save(checkpoint_path, checkpoint)
        cache.atomic_json_write(
            root / f"{arm}_summary.json",
            {
                "arm": artifact_arm,
                "best_epoch": 1,
                "best_dev_metrics": {"average_precision": ap},
                "sidecar_checkpoint": str(checkpoint_path),
                "sidecar_checkpoint_sha256": cache.sha256_file(checkpoint_path),
                "combined_initial_state_sha256": "shared-fresh-initial-state",
                "test_or_sealed_or_holdout_read": False,
            },
        )
        return checkpoint_path

    @staticmethod
    def _prediction_frame(probability: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [f"row{i}" for i in range(8)],
                "plume_id": [f"event{i // 2}-x" for i in range(8)],
                "event_id": [f"event{i // 2}" for i in range(8)],
                "label": [0, 0, 0, 1, 0, 1, 1, 1],
                "probability": probability,
            }
        )

    def _write_predictions(self, root: Path) -> dict[str, pd.DataFrame]:
        probabilities = {
            "p0": [0.10, 0.15, 0.35, 0.65, 0.40, 0.75, 0.80, 0.85],
            "p4": [0.12, 0.18, 0.32, 0.62, 0.38, 0.72, 0.78, 0.82],
            "p5": [0.11, 0.16, 0.30, 0.70, 0.35, 0.78, 0.82, 0.88],
            "d1_seed_20260727": [
                0.08,
                0.12,
                0.28,
                0.72,
                0.30,
                0.80,
                0.85,
                0.90,
            ],
            "d1_seed_20260728": [
                0.09,
                0.13,
                0.29,
                0.71,
                0.31,
                0.79,
                0.84,
                0.89,
            ],
            "d1_seed_20260729": [
                0.07,
                0.11,
                0.27,
                0.73,
                0.29,
                0.81,
                0.86,
                0.91,
            ],
        }
        frames = {
            name: self._prediction_frame(value)
            for name, value in probabilities.items()
        }
        for arm in ("p0", "p4", "p5"):
            frames[arm]["arm"] = arm
            frames[arm]["epoch"] = 1
        cache.atomic_csv_write(root / "p0.csv", frames["p0"])
        cache.atomic_csv_write(root / "p0_replay.csv", frames["p0"])
        cache.atomic_csv_write(root / "p4.csv", frames["p4"])
        cache.atomic_csv_write(root / "p5.csv", frames["p5"])
        for seed in target.FROZEN_SEEDS:
            seed_dir = root / "d1" / f"seed_{seed}"
            seed_dir.mkdir(parents=True)
            cache.atomic_csv_write(
                seed_dir / "p0_base_predictions.csv", frames["p0"]
            )
            cache.atomic_csv_write(
                seed_dir / "d1.csv", frames[f"d1_seed_{seed}"]
            )
        return frames

    @staticmethod
    def _head_metric(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "best_epoch": 1,
            **target.recompute_head_metric_bundle(frame),
        }

    @staticmethod
    def _write_p0_parent(root: Path, p0_frame: pd.DataFrame) -> Path:
        parent_dir = root / "p0"
        parent_dir.mkdir()
        checkpoint_path = (
            parent_dir / "checkpoint_best_event_balanced_ap.pt"
        )
        metrics = target.tempo.metric_bundle(
            p0_frame["label"].to_numpy(dtype=np.int64),
            p0_frame["probability"].to_numpy(dtype=np.float64),
            p0_frame["event_id"].astype(str).tolist(),
        )
        cache.atomic_torch_save(
            checkpoint_path,
            {
                "arm": "p0",
                "epoch": 1,
                "model": {"weight": torch.zeros(1)},
                "validation": metrics,
                "test_or_sealed_or_holdout_read": False,
            },
        )
        return checkpoint_path

    @staticmethod
    def _write_d1_summary(
        path: Path,
        seed: int,
        p0_frame: pd.DataFrame,
        d1_frame: pd.DataFrame,
        p0_checkpoint: Path,
        p0_prediction: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = p0_frame["label"].to_numpy(dtype=np.int64)
        events = p0_frame["event_id"].astype(str).tolist()
        p0_metrics = target.tempo.metric_bundle(
            labels,
            p0_frame["probability"].to_numpy(dtype=np.float64),
            events,
        )
        d1_metrics = target.tempo.metric_bundle(
            labels,
            d1_frame["probability"].to_numpy(dtype=np.float64),
            events,
        )
        shuffled = dict(d1_metrics)
        shuffled["event_balanced_ap"] -= 0.01
        shuffled["event_balanced_auc"] -= 0.01
        parent_payload = cache.torch_load_trusted(p0_checkpoint)
        parent_model_sha = cache.state_dict_sha256(parent_payload["model"])
        parent_audit = {
            "base_kind": "event_balanced_p0",
            "checkpoint": str(p0_checkpoint),
            "checkpoint_sha256": cache.sha256_file(p0_checkpoint),
            "checkpoint_epoch": 1,
            "checkpoint_validation": p0_metrics,
            "prediction_csv": str(p0_prediction),
            "prediction_csv_sha256": cache.sha256_file(p0_prediction),
            "replay_max_abs_probability_error": 0.0,
            "replay_tolerance": 1e-6,
            "replay_numerically_exact": True,
            "model_state_sha256": parent_model_sha,
            "base_feature_contract": (
                "[original_Panopticon_768, exact_zero_768]"
            ),
            "response_sidecar_used": False,
            "test_or_sealed_or_holdout_read": False,
        }
        history = [
            {
                "arm": "d1_gated_delta",
                "epoch": 1,
                "optimizer_steps": 1,
                "train_loss": 0.5,
                "classification_loss": 0.5,
                "null_loss": 0.0,
                "validation": d1_metrics,
                "elapsed_seconds": 1.0,
            },
            {
                "arm": "d1_gated_delta",
                "epoch": 2,
                "optimizer_steps": 1,
                "train_loss": 0.5,
                "classification_loss": 0.5,
                "null_loss": 0.0,
                "validation": d1_metrics,
                "elapsed_seconds": 1.0,
            },
        ]
        history_path = path.parent / "d1_gated_delta_metrics_history.json"
        cache.atomic_json_write(history_path, history)
        temporal_initial_sha = f"temporal-initial-{seed}"
        zero_init_replay = {
            "evaluated_before_any_temporal_optimizer_step": True,
            "arm": "d1_gated_delta",
            "zero_initialized_residual_output": True,
            "fresh_p0_checkpoint": str(p0_checkpoint),
            "fresh_p0_checkpoint_sha256": cache.sha256_file(p0_checkpoint),
            "fresh_p0_model_state_sha256": parent_model_sha,
            "temporal_initial_state_sha256": temporal_initial_sha,
            "rows": int(len(p0_frame)),
            "maximum_absolute_probability_error": 0.0,
            "probability_tolerance": 1e-7,
            "maximum_absolute_logit_error": 0.0,
            "logit_tolerance": 1e-6,
            "pass": True,
        }
        d1_checkpoint_path = (
            path.parent / "d1_gated_delta_best_event_ap.pt"
        )
        cache.atomic_torch_save(
            d1_checkpoint_path,
            {
                "arm": "d1_gated_delta",
                "seed": seed,
                "epoch": 1,
                "model": {"weight": torch.zeros(1)},
                "initial_state_sha256": temporal_initial_sha,
                "frozen_base_checkpoint": str(p0_checkpoint),
                "frozen_base_checkpoint_sha256": cache.sha256_file(
                    p0_checkpoint
                ),
                "frozen_base_model_state_sha256": parent_model_sha,
                "validation": d1_metrics,
                "test_or_sealed_or_holdout_read": False,
            },
        )
        shuffle_delta = {
            key: float(shuffled[key] - d1_metrics[key])
            for key in target.GATE_METRICS
        }
        cache.atomic_json_write(
            path,
            {
                "script_version": target.tempo.SCRIPT_VERSION,
                "seed": seed,
                "frozen_role_only_base_audit": parent_audit,
                "base_initial_state_sha256": parent_model_sha,
                "temporal_initial_state_sha256": temporal_initial_sha,
                "zero_initialized_d1_prototype_exact_p0_replay": (
                    zero_init_replay
                ),
                "results": {
                    "p0_base": {
                        "arm": "p0_base",
                        "seed": seed,
                        "best": {"validation": p0_metrics},
                        "epoch_zero_exact_frozen_base": {
                            "replay_max_abs_probability_error": 0.0,
                            "replay_tolerance": 1e-6,
                            "exact_within_tolerance": True,
                            "source_checkpoint_epoch": 1,
                        },
                        "frozen_role_only_base_audit": parent_audit,
                        "zero_initialized_d1_prototype_exact_p0_replay": (
                            zero_init_replay
                        ),
                    },
                    "d1_gated_delta": {
                        "arm": "d1_gated_delta",
                        "seed": seed,
                        "best": {"epoch": 1, "validation": d1_metrics},
                        "initial_state_sha256": temporal_initial_sha,
                        "epoch_zero_exact_p0_replay": zero_init_replay,
                        "early_stop_receipt": {
                            "selection_metric": "event_balanced_ap",
                            "selection_tie_rule": (
                                "earliest epoch attaining maximum AP"
                            ),
                            "max_epochs": 4,
                            "patience": 1,
                            "observed_epochs": [1, 2],
                            "observed_event_balanced_ap": [
                                d1_metrics["event_balanced_ap"],
                                d1_metrics["event_balanced_ap"],
                            ],
                            "epochs_observed": 2,
                            "selected_epoch": 1,
                            "selected_event_balanced_ap": d1_metrics[
                                "event_balanced_ap"
                            ],
                            "stop_reason": "patience_exhausted",
                            "history_path": str(history_path.resolve()),
                            "history_sha256": cache.sha256_file(history_path),
                            "valid": True,
                        },
                        "history_shuffle_fixed_model_and_threshold": shuffled,
                        "history_shuffle_delta": shuffle_delta,
                        "history_shuffle_validity": {
                            "seed": seed + 8_191,
                            "rows": int(len(p0_frame)),
                            "history_indices": [1, 2, 3, 4, 5],
                            "strata_count": 1,
                            "pattern_count": 1,
                            "strata": [
                                {
                                    "stratum_index": 0,
                                    "valid_history_pattern": [
                                        True,
                                        True,
                                        True,
                                        True,
                                        True,
                                    ],
                                    "unique_history_pattern": [
                                        True,
                                        True,
                                        True,
                                        True,
                                        True,
                                    ],
                                    "rows": int(len(p0_frame)),
                                    "canonical_event_count": int(
                                        p0_frame["event_id"].nunique()
                                    ),
                                    "donor_indices_sha256": "b" * 64,
                                    "all_cross_event": True,
                                }
                            ],
                            "failed_strata": [],
                            "availability_mismatch_rows": 0,
                            "availability_pattern_mismatch_count": 0,
                            "all_target_donor_strata_equal": True,
                            "all_cross_event": True,
                            "donor_indices_sha256": "a" * 64,
                            "all_donors_from_different_canonical_event": True,
                            "ordered_id_preserved": True,
                            "ordered_plume_id_preserved": True,
                            "ordered_event_id_preserved": True,
                            "labels_preserved": True,
                            "shapes_preserved": True,
                            "t0_feature_preserved": True,
                            "availability_pattern_preserved": True,
                            "acquisition_metadata_preserved": True,
                            "only_history_features_replaced": True,
                            "valid": True,
                        },
                    },
                },
                "test_or_sealed_or_holdout_read": False,
            },
        )

    @staticmethod
    def _write_fixed_result(
        path: Path,
        root: Path,
        frames: dict[str, pd.DataFrame],
    ) -> None:
        labels = frames["p5"]["label"].to_numpy(dtype=np.int64)
        events = frames["p5"]["event_id"].astype(str).tolist()
        d1 = [
            frames[f"d1_seed_{seed}"]["probability"].to_numpy(dtype=np.float64)
            for seed in target.FROZEN_SEEDS
        ]
        mean_d1, candidate = target.fixed_reference_mean_d1(
            frames["p5"]["probability"].to_numpy(dtype=np.float64),
            d1,
        )
        probabilities = {
            "p0": frames["p0"]["probability"].to_numpy(dtype=np.float64),
            "p5": frames["p5"]["probability"].to_numpy(dtype=np.float64),
            "mean_seed_d1": mean_d1,
            "fixed_p5_mean_d1": candidate,
        }
        points = {
            name: target.tempo.metric_bundle(labels, values, events)
            for name, values in probabilities.items()
        }
        bootstrap = fake_bootstrap(
            labels,
            events,
            probabilities,
            points,
            (
                ("candidate_minus_p5", "fixed_p5_mean_d1", "p5"),
                ("candidate_minus_p0", "fixed_p5_mean_d1", "p0"),
            ),
            replicates=5000,
            seed=2026072808,
        )
        candidate_path = root / "fixed_p5_candidate.csv"
        candidate_frame = frames["p5"].copy()
        candidate_frame["probability"] = candidate
        cache.atomic_csv_write(candidate_path, candidate_frame)
        provenance = {
            "p0": root / "p0_replay.csv",
            "p5": root / "p5.csv",
            **{
                f"d1_seed_{seed}": root / "d1" / f"seed_{seed}" / "d1.csv"
                for seed in target.FROZEN_SEEDS
            },
        }
        cache.atomic_json_write(
            path,
            {
                "schema_version": "l89-clean-fixed-p5-mean-d1-v1",
                "arithmetic": {
                    "d1_seeds": list(target.FROZEN_SEEDS),
                    "p5_logit_weight": 0.5,
                    "mean_d1_logit_weight": 0.5,
                    "weights_refit": False,
                },
                "bootstrap": {
                    "replicates": 5000,
                    "seed": 2026072808,
                    "point_selected_thresholds_fixed_per_system": True,
                    "thresholds_refit_per_replicate": False,
                },
                "point_metrics": points,
                "paired_event_cluster_bootstrap": bootstrap,
                "prediction_output": {
                    "path": str(candidate_path),
                    "sha256": cache.sha256_file(candidate_path),
                },
                "input_provenance": {
                    name: {
                        "path": str(value),
                        "sha256": cache.sha256_file(value),
                    }
                    for name, value in provenance.items()
                },
                "test_or_sealed_or_holdout_or_outer_read": False,
                "formal_inner_development_only": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
