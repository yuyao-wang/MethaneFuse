#!/usr/bin/env python3
"""Dataset-free CPU regression tests for the authorized RankNet fallback."""

from __future__ import annotations

import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import compare_ranknet_scratch as ranknet_comparator
import evaluate_ranknet_validation as evaluator
import multisensor_residual_runner as runner


def resume_args(*, ranknet: bool) -> Namespace:
    return Namespace(
        mode="supervised",
        sharing="shared",
        mask_ratio=0.6,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=0.05,
        grad_clip=1.0,
        balanced_pos_weight=True,
        amp=False,
        augment=True,
        seed=20260727,
        max_val_batches=0,
        validity_masked_reconstruction=False,
        ranknet_objective=ranknet,
    )


def validation_args(**updates: Any) -> Namespace:
    values = {
        "ranknet_objective": True,
        "ranknet_weight": runner.RANKNET_FIXED_WEIGHT,
        "ranknet_temperature": runner.RANKNET_FIXED_TEMPERATURE,
        "mode": "supervised",
        "sharing": "shared",
        "sensors": ",".join(runner.SENSOR_ORDER),
        "balanced_pos_weight": True,
        "validity_masked_reconstruction": False,
        "init_checkpoint": None,
        "resume": None,
        "no_save_checkpoint": False,
        "epochs": 1,
        "seed": 20260727,
        "augment": True,
    }
    values.update(updates)
    return Namespace(**values)


class FakeSensorModel(nn.Module):
    """Minimal differentiable model used to test train-loop pair isolation."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        sensor: str,
        streams: Mapping[str, torch.Tensor],
        validity_by_stream=None,
    ) -> Dict[str, torch.Tensor]:
        del sensor, validity_by_stream
        logits = streams["value"].reshape(streams["value"].shape[0]) * self.scale
        return {"logits": logits}


class RankNetLossTests(unittest.TestCase):
    def test_01_hand_computed_all_pairs(self) -> None:
        logits = torch.tensor([2.0, -1.0, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        observed, diagnostics = runner.all_pairs_ranknet_loss(
            logits,
            labels,
            temperature=1.0,
        )
        positive = [2.0, 0.5]
        negative = [-1.0, -0.5]
        expected_terms = [
            F.softplus(torch.tensor(negative_logit - positive_logit))
            for positive_logit in positive
            for negative_logit in negative
        ]
        expected = torch.stack(expected_terms).mean()
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(diagnostics["positive_samples"], 2)
        self.assertEqual(diagnostics["negative_samples"], 2)
        self.assertEqual(diagnostics["pair_count"], 4)
        self.assertEqual(diagnostics["one_class_batch"], 0)

    def test_02_permutation_invariance(self) -> None:
        logits = torch.tensor([1.2, -0.7, 0.3, 2.1, -1.4])
        labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        permutation = torch.tensor([3, 1, 4, 0, 2])
        original, original_diagnostics = runner.all_pairs_ranknet_loss(
            logits,
            labels,
            temperature=1.0,
        )
        permuted, permuted_diagnostics = runner.all_pairs_ranknet_loss(
            logits[permutation],
            labels[permutation],
            temperature=1.0,
        )
        self.assertTrue(torch.equal(original, permuted))
        self.assertEqual(original_diagnostics, permuted_diagnostics)

    def test_03_gradient_direction_and_shift_invariance(self) -> None:
        logits = torch.tensor([0.2, -0.3], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        loss, _ = runner.all_pairs_ranknet_loss(
            logits,
            labels,
            temperature=1.0,
        )
        loss.backward()
        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertGreater(float(logits.grad[1]), 0.0)
        shifted, _ = runner.all_pairs_ranknet_loss(
            logits.detach() + 123.0,
            labels,
            temperature=1.0,
        )
        self.assertTrue(torch.allclose(loss.detach(), shifted, atol=1e-6))

    def test_04_extreme_logits_are_finite(self) -> None:
        for values in ([100.0, -100.0], [-100.0, 100.0]):
            logits = torch.tensor(values, requires_grad=True)
            labels = torch.tensor([1.0, 0.0])
            loss, _ = runner.all_pairs_ranknet_loss(
                logits,
                labels,
                temperature=1.0,
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_05_one_class_connected_zero_and_bce_backward(self) -> None:
        classification_loss = nn.BCEWithLogitsLoss()
        for label_value in (0.0, 1.0):
            logits = torch.tensor([0.2, -0.1, 0.7], requires_grad=True)
            labels = torch.full((3,), label_value)
            rank_loss, diagnostics = runner.all_pairs_ranknet_loss(
                logits,
                labels,
                temperature=1.0,
            )
            self.assertEqual(float(rank_loss), 0.0)
            self.assertEqual(diagnostics["pair_count"], 0)
            self.assertEqual(diagnostics["one_class_batch"], 1)
            total, _ = runner.supervised_classification_objective(
                logits,
                labels,
                classification_loss,
                ranknet_enabled=True,
                ranknet_weight=0.5,
                ranknet_temperature=1.0,
            )
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_06_train_loop_keeps_sensor_pairs_isolated(self) -> None:
        model = FakeSensorModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_loaders = {
            "sensor_a": [
                (
                    {
                        "value": torch.tensor(
                            [[[[1.0]]], [[[-1.0]]]]
                        )
                    },
                    torch.tensor([1.0, 0.0]),
                )
            ],
            "sensor_b": [
                (
                    {
                        "value": torch.tensor(
                            [[[[2.0]]], [[[-2.0]]]]
                        )
                    },
                    torch.tensor([1.0, 0.0]),
                )
            ],
        }
        losses = {
            sensor: nn.BCEWithLogitsLoss() for sensor in train_loaders
        }
        _, _, diagnostics = runner.train_one_epoch(
            model,
            train_loaders,
            optimizer,
            losses,
            sensors=("sensor_a", "sensor_b"),
            device=torch.device("cpu"),
            amp=False,
            grad_clip=0.0,
            rounds=1,
            mode="supervised",
            image_size=1,
            log_interval_rounds=0,
            validity_masked_reconstruction=False,
            ranknet_enabled=True,
            ranknet_weight=0.5,
            ranknet_temperature=1.0,
        )
        self.assertEqual(diagnostics["sensor_a"]["pair_count"], 1)
        self.assertEqual(diagnostics["sensor_b"]["pair_count"], 1)
        self.assertEqual(diagnostics["sensor_a"]["batches"], 1)
        self.assertEqual(diagnostics["sensor_b"]["batches"], 1)
        self.assertEqual(
            diagnostics["sensor_a"]["paired_batch_fraction"],
            1.0,
        )
        self.assertEqual(
            diagnostics["sensor_a"]["one_class_batch_fraction"],
            0.0,
        )

    def test_07_zero_weight_matches_legacy_bce_exactly(self) -> None:
        logits = torch.tensor([0.2, -0.5, 1.1, -1.7])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        classification_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(1.3)
        )
        expected = classification_loss(logits, labels)
        observed, _ = runner.supervised_classification_objective(
            logits,
            labels,
            classification_loss,
            ranknet_enabled=True,
            ranknet_weight=0.0,
            ranknet_temperature=1.0,
        )
        disabled, _ = runner.supervised_classification_objective(
            logits,
            labels,
            classification_loss,
            ranknet_enabled=False,
            ranknet_weight=0.5,
            ranknet_temperature=1.0,
        )
        self.assertTrue(torch.equal(expected, observed))
        self.assertTrue(torch.equal(expected, disabled))


class RankNetIntegrationTests(unittest.TestCase):
    def test_08_tiny_s2_s5p_forward_backward(self) -> None:
        torch.manual_seed(31)
        model = runner.MultiSensorResidualModel(
            ("s2", "s5p"),
            mode="supervised",
            sharing="shared",
            image_size=28,
            patch_size=14,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            fuse_freq=1,
            dropout=0.0,
            mask_ratio=0.5,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=4,
            validity_masked_reconstruction=False,
        )
        total = None
        for sensor in ("s2", "s5p"):
            streams = {
                f"{sensor}_{suffix}": torch.randn(
                    4,
                    runner.SENSOR_SPECS[sensor].channels,
                    28,
                    28,
                )
                for suffix in runner.STREAM_SUFFIXES
            }
            labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
            logits = model(sensor, streams)["logits"]
            loss, _ = runner.supervised_classification_objective(
                logits,
                labels,
                nn.BCEWithLogitsLoss(),
                ranknet_enabled=True,
                ranknet_weight=0.5,
                ranknet_temperature=1.0,
            )
            total = loss if total is None else total + loss
        assert total is not None
        total.backward()
        backbone_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if (
                name.startswith("backbone.")
                and parameter.requires_grad
                and parameter.grad is not None
            )
        ]
        self.assertTrue(backbone_gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in backbone_gradients)
        )
        for sensor in ("s2", "s5p"):
            gradients = [
                parameter.grad
                for parameter in model.heads[sensor].parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )

    def test_09_objective_and_resume_signature_contract(self) -> None:
        encoder_signature = {"encoder": "same"}
        data_signature = {"data": "same"}
        legacy = runner.build_resume_signature(
            resume_args(ranknet=False),
            runner.SENSOR_ORDER,
            encoder_signature,
            data_signature,
            rounds=157,
            loader_lengths={sensor: 200 for sensor in runner.SENSOR_ORDER},
        )
        ranknet = runner.build_resume_signature(
            resume_args(ranknet=True),
            runner.SENSOR_ORDER,
            encoder_signature,
            data_signature,
            rounds=157,
            loader_lengths={sensor: 200 for sensor in runner.SENSOR_ORDER},
        )
        self.assertEqual(legacy["schema_version"], 1)
        self.assertNotIn("supervised_objective", legacy)
        self.assertEqual(ranknet["schema_version"], 2)
        self.assertEqual(
            ranknet["supervised_objective"],
            runner.build_ranknet_objective_signature(),
        )
        self.assertNotEqual(legacy, ranknet)
        with self.assertRaises(ValueError):
            runner.validate_ranknet_args(
                validation_args(ranknet_weight=0.4),
                runner.SENSOR_ORDER,
            )
        with self.assertRaises(ValueError):
            runner.validate_ranknet_args(
                validation_args(ranknet_temperature=0.9),
                runner.SENSOR_ORDER,
            )
        with self.assertRaises(ValueError):
            runner.validate_ranknet_args(
                validation_args(seed=7),
                runner.SENSOR_ORDER,
            )

    def test_10_matched_comparator_audit(self) -> None:
        # The comparator's dataset-free synthetic suite mutates one contract
        # field at a time and proves that only the preregistered objective
        # delta is allowed.
        self.assertIsNone(ranknet_comparator.run_self_test())

    def test_11_metrics_recompute_from_predictions(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int64)
        probabilities = np.asarray(
            [0.9, 0.1, 0.8, 0.4, 0.3, 0.2],
            dtype=np.float64,
        )
        metrics = runner.classification_metrics(labels, probabilities)
        self.assertAlmostEqual(
            metrics["ap"],
            average_precision_score(labels, probabilities),
            places=15,
        )
        self.assertAlmostEqual(
            metrics["auroc"],
            roc_auc_score(labels, probabilities),
            places=15,
        )
        predictions = probabilities >= float(metrics["threshold"])
        true_positive = int(((labels == 1) & predictions).sum())
        false_positive = int(((labels == 0) & predictions).sum())
        false_negative = int(((labels == 1) & ~predictions).sum())
        expected_f1 = (
            2 * true_positive
            / (2 * true_positive + false_positive + false_negative)
        )
        self.assertAlmostEqual(metrics["f1"], expected_f1, places=15)

    def test_12_current_only_zeroes_values_and_validity(self) -> None:
        streams = {
            "s2_current": torch.ones(2, 1, 2, 2),
            "s2_recent": torch.full((2, 1, 2, 2), 2.0),
            "s2_seasonal": torch.full((2, 1, 2, 2), 3.0),
        }
        validity = {
            key: torch.ones_like(value)
            for key, value in streams.items()
        }
        current_only, current_validity = evaluator.apply_input_contract(
            streams,
            validity,
            "current_only",
        )
        self.assertTrue(
            torch.equal(current_only["s2_current"], streams["s2_current"])
        )
        self.assertEqual(float(current_only["s2_recent"].abs().sum()), 0.0)
        self.assertEqual(float(current_only["s2_seasonal"].abs().sum()), 0.0)
        assert current_validity is not None
        self.assertTrue(
            torch.equal(
                current_validity["s2_current"],
                validity["s2_current"],
            )
        )
        self.assertEqual(
            float(current_validity["s2_recent"].abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(current_validity["s2_seasonal"].abs().sum()),
            0.0,
        )
        full, full_validity = evaluator.apply_input_contract(
            streams,
            validity,
            "full_temporal",
        )
        for key in streams:
            self.assertTrue(torch.equal(full[key], streams[key]))
            assert full_validity is not None
            self.assertTrue(torch.equal(full_validity[key], validity[key]))

        torch.manual_seed(41)
        model = runner.MultiSensorResidualModel(
            ("s5p",),
            mode="supervised",
            sharing="shared",
            image_size=28,
            patch_size=14,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            fuse_freq=1,
            dropout=0.0,
            mask_ratio=0.5,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=4,
            validity_masked_reconstruction=False,
        ).eval()
        model_streams = {
            f"s5p_{suffix}": torch.randn(2, 1, 28, 28)
            for suffix in runner.STREAM_SUFFIXES
        }
        ablated, _ = evaluator.apply_input_contract(
            model_streams,
            None,
            "current_only",
        )
        with torch.inference_mode():
            full_logits = model("s5p", model_streams)["logits"]
            current_logits = model("s5p", ablated)["logits"]
        self.assertTrue(torch.isfinite(full_logits).all())
        self.assertTrue(torch.isfinite(current_logits).all())

    def test_13_full_validation_artifact_writer(self) -> None:
        objective = runner.build_ranknet_objective_signature()
        per_sensor = {
            sensor: {
                "samples": 1,
                "positive_rate": float(index % 2),
                "ap": 1.0,
                "auroc": 0.5,
                "f1": float(index % 2),
                "f1_0p5": float(index % 2),
                "threshold": 0.5,
                "probability_min": 0.25 + index * 0.1,
                "probability_max": 0.25 + index * 0.1,
                "probability_mean": 0.25 + index * 0.1,
                "probability_std": 0.0,
                "predicted_positive_at_0p5": int(index >= 3),
                "predicted_positive_at_best": int(index >= 3),
            }
            for index, sensor in enumerate(runner.SENSOR_ORDER)
        }
        prediction_rows = [
            {
                "sensor": sensor,
                "row_index": 0,
                "label": int(index % 2),
                "probability": 0.25 + index * 0.1,
            }
            for index, sensor in enumerate(runner.SENSOR_ORDER)
        ]
        with tempfile.TemporaryDirectory(
            prefix="ranknet_artifact_writer_test."
        ) as directory:
            root = Path(directory)
            history_path = root / "metrics_history.json"
            checkpoint_path = root / "checkpoint_latest.pth"
            history = {
                "status": "completed",
                "sensors": list(runner.SENSOR_ORDER),
                "normalization_stats": str(root / "stats.json"),
                "supervised_objective": objective,
                "encoder_signature": {"encoder": "v1"},
                "data_signature": {
                    "event_protocol_fingerprint": "protocol-v1",
                    "normalization_stats_sha256": "stats-sha",
                },
                "resume_signature": {
                    "schema_version": 2,
                    "supervised_objective": objective,
                },
            }
            runner.atomic_json_dump(history, history_path)
            runner.atomic_torch_save({"epoch": 0}, checkpoint_path)
            report = runner.write_ranknet_full_validation_artifacts(
                history=history,
                history_path=history_path,
                checkpoint_path=checkpoint_path,
                output_dir=root,
                epoch=0,
                per_sensor=per_sensor,
                macro_over_sensor={
                    "ap": 1.0,
                    "auroc": 0.5,
                    "f1": 0.5,
                    "f1_0p5": 0.5,
                },
                prediction_rows=prediction_rows,
                runtime={"device": "cpu"},
            )
            output_json = root / "validation_full_temporal.json"
            output_csv = root / "validation_full_temporal.csv"
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_csv.is_file())
            payload = evaluator.load_json_mapping(output_json)
            self.assertEqual(
                payload["artifact_type"],
                runner.RANKNET_VALIDATION_ARTIFACT_TYPE,
            )
            self.assertEqual(
                payload["input_contract"],
                runner.RANKNET_FULL_INPUT_CONTRACT,
            )
            self.assertEqual(payload["prediction_rows"], 4)
            self.assertEqual(
                payload["predictions_csv_sha256"],
                runner.sha256_file(output_csv),
            )
            self.assertEqual(
                report["checkpoint_sha256"],
                runner.sha256_file(checkpoint_path),
            )

    def test_14_legacy_artifact_schema_is_unchanged(self) -> None:
        legacy_config = runner.artifact_config(
            Namespace(
                mode="supervised",
                ranknet_objective=False,
                ranknet_weight=0.5,
                ranknet_temperature=1.0,
            )
        )
        self.assertEqual(legacy_config, {"mode": "supervised"})
        self.assertEqual(
            runner.EPOCH_TABLE_FIELDS,
            (
                "epoch",
                "sensor",
                "mode",
                "sharing",
                "train_loss",
                "val_loss",
                "reconstruction_loss",
                "samples",
                "positive_rate",
                "ap",
                "f1",
                "f1_0p5",
                "auroc",
                "threshold",
                "elapsed_seconds",
                "completed_unix",
            ),
        )

    def test_15_evaluator_accepts_signed_nested_stats(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="ranknet_evaluator_inputs_test."
        ) as directory:
            root = Path(directory)
            resolved: Dict[str, Dict[str, str]] = {}
            manifest_hashes: Dict[str, Dict[str, str]] = {}
            for sensor in runner.SENSOR_ORDER:
                resolved[sensor] = {}
                manifest_hashes[sensor] = {}
                for split in ("train", "val"):
                    path = root / f"{sensor}_{split}.csv"
                    runner.atomic_text_dump("label\n0\n", path)
                    resolved[sensor][split] = str(path)
                    manifest_hashes[sensor][split] = runner.sha256_file(path)
            stats_path = root / "stats.json"
            runner.atomic_json_dump(
                {
                    "schema_version": 2,
                    "sensors": {
                        sensor: {
                            "mean": [
                                0.0
                                for _ in range(
                                    runner.SENSOR_SPECS[sensor].channels
                                )
                            ],
                            "std": [
                                1.0
                                for _ in range(
                                    runner.SENSOR_SPECS[sensor].channels
                                )
                            ],
                        }
                        for sensor in runner.SENSOR_ORDER
                    },
                },
                stats_path,
            )
            history = {
                "resolved_csvs": resolved,
                "normalization_stats": str(stats_path),
                "data_signature": {
                    "manifest_sha256": manifest_hashes,
                    "normalization_stats_sha256": runner.sha256_file(
                        stats_path
                    ),
                },
            }
            csvs, observed_stats_path, stats = (
                evaluator.verify_signed_inputs(
                    history,
                    runner.SENSOR_ORDER,
                )
            )
            self.assertEqual(observed_stats_path, stats_path)
            self.assertEqual(set(csvs), set(runner.SENSOR_ORDER))
            self.assertEqual(set(stats), set(runner.SENSOR_ORDER))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
