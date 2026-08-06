#!/usr/bin/env python3
"""CPU regression tests for the 360 m hierarchical TransientQuery head."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import query360_model as tq


class Query360ModelTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(11)
        features = torch.randn(8, 4, 3, 16, generator=generator)
        valid = torch.tensor(
            [
                [[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1]],
                [[1, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ],
            dtype=torch.bool,
        )
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
        return features, valid, labels

    @staticmethod
    def _model() -> tq.TransientQuery360Head:
        return tq.TransientQuery360Head(
            feature_dim=16,
            model_dim=32,
            num_heads=4,
            depth=2,
            dropout=0.0,
        )

    def test_01_shapes_masks_and_backward(self) -> None:
        features, valid, labels = self._inputs()
        model = self._model()
        model.train()
        output = model(features, valid, arm="transient_query")
        self.assertEqual(tuple(output.fused_logits.shape), (8,))
        self.assertEqual(tuple(output.sensor_logits.shape), (8, 4))
        self.assertEqual(tuple(output.sensor_evidence.shape), (8, 4, 32))
        self.assertEqual(tuple(output.sensor_attention.shape), (8, 4))
        self.assertTrue(torch.equal(output.sensor_valid, valid[:, :, 0]))
        self.assertTrue(torch.equal(output.effective_time_valid, valid))
        self.assertTrue(torch.isfinite(output.fused_logits).all())
        self.assertTrue(torch.isfinite(output.sensor_logits).all())
        self.assertTrue(
            torch.allclose(
                output.sensor_attention.sum(dim=1), torch.ones(8), atol=1e-6
            )
        )

        invalid = ~valid[:, :, 0]
        self.assertEqual(int(torch.count_nonzero(output.sensor_attention[invalid])), 0)
        self.assertEqual(int(torch.count_nonzero(output.sensor_logits[invalid])), 0)
        self.assertEqual(int(torch.count_nonzero(output.sensor_evidence[invalid])), 0)

        loss = tq.transient_query_loss(output, labels)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertEqual(loss.valid_sensor_count, int(valid[:, :, 0].sum()))
        loss.total.backward()
        self.assertIsNotNone(model.fused_classifier.weight.grad)
        self.assertGreater(
            float(model.fused_classifier.weight.grad.abs().sum()), 0.0
        )

    def test_02_current_only_masks_history_but_has_matched_parameters(self) -> None:
        features, valid, _ = self._inputs()
        models, initialization_hash, signature = tq.build_matched_models(
            {
                "feature_dim": 16,
                "model_dim": 32,
                "num_heads": 4,
                "depth": 2,
                "dropout": 0.0,
            },
            seed=23,
        )
        self.assertEqual(set(models), set(tq.ARM_NAMES))
        self.assertEqual(len(initialization_hash), 64)
        hashes = {
            tq.state_dict_sha256(model.state_dict()) for model in models.values()
        }
        self.assertEqual(hashes, {initialization_hash})
        signatures = {
            tq.model_parameter_signature(model)["shape_sha256"]
            for model in models.values()
        }
        self.assertEqual(signatures, {signature["shape_sha256"]})

        current = models["current_only"].eval()(
            features, valid, arm="current_only"
        )
        self.assertTrue(current.effective_time_valid[:, :, 0].equal(valid[:, :, 0]))
        self.assertFalse(current.effective_time_valid[:, :, 1:].any())

        # Changing masked history cannot change current-only predictions.
        altered = features.clone()
        altered[:, :, 1:] += 10_000
        changed = models["current_only"].eval()(
            altered, valid, arm="current_only"
        )
        self.assertTrue(
            torch.allclose(current.fused_logits, changed.fused_logits, atol=0, rtol=0)
        )

        scale_aware = models["scale_aware_transient_query"].eval()(
            features, valid, arm="scale_aware_transient_query"
        )
        self.assertTrue(
            scale_aware.effective_time_valid[:, :3].equal(valid[:, :3])
        )
        self.assertTrue(
            scale_aware.effective_time_valid[:, 3, 0].equal(valid[:, 3, 0])
        )
        self.assertFalse(
            scale_aware.effective_time_valid[:, 3, 1:].any()
        )

    def test_03_shuffle_is_per_sensor_cross_event_and_preserves_mask(self) -> None:
        rows, sensors, roles, dim = 6, 4, 3, 16
        features = torch.arange(
            rows * sensors * roles * dim, dtype=torch.float32
        ).reshape(rows, sensors, roles, dim)
        valid = torch.ones(rows, sensors, roles, dtype=torch.bool)
        # A missing role forces donor construction to respect target patterns.
        valid[0, 2, 2] = False
        events = [f"event-{index}" for index in range(rows)]
        plumes = [f"plume-{index}" for index in range(rows)]
        donors_a = tq.build_history_shuffle_donors(
            valid, events, plume_ids=plumes, seed=31
        )
        donors_b = tq.build_history_shuffle_donors(
            valid, events, plume_ids=plumes, seed=31
        )
        self.assertTrue(torch.equal(donors_a, donors_b))
        for row in range(rows):
            for sensor in range(sensors):
                donor = int(donors_a[row, sensor])
                self.assertGreaterEqual(donor, 0)
                self.assertNotEqual(events[row], events[donor])
                self.assertNotEqual(plumes[row], plumes[donor])
                target_history = valid[row, sensor, 1:]
                self.assertTrue(valid[donor, sensor, 1:][target_history].all())

        shuffled = tq.apply_history_shuffle(
            features,
            valid,
            donors_a,
            event_ids=events,
            plume_ids=plumes,
        )
        self.assertTrue(torch.equal(shuffled[:, :, 0], features[:, :, 0]))
        for row in range(rows):
            for sensor in range(sensors):
                donor = int(donors_a[row, sensor])
                for role in range(1, roles):
                    if valid[row, sensor, role]:
                        self.assertTrue(
                            torch.equal(
                                shuffled[row, sensor, role],
                                features[donor, sensor, role],
                            )
                        )

        # Validation of the shuffle arm must use coherent evidence.
        model = self._model().eval()
        coherent = model(features, valid, arm="transient_query")
        shuffle_eval = model(
            features,
            valid,
            arm="history_shuffle_train",
            donor_indices=donors_a,
        )
        self.assertTrue(
            torch.equal(coherent.fused_logits, shuffle_eval.fused_logits)
        )

    def test_04_invalid_sensors_never_affect_fusion_or_auxiliary_loss(self) -> None:
        features, valid, labels = self._inputs()
        model = self._model().eval()
        baseline = model(features, valid, arm="transient_query")
        altered = features.clone()
        invalid = ~valid[:, :, 0]
        altered[invalid] = torch.randn_like(altered[invalid]) * 1_000_000
        candidate = model(altered, valid, arm="transient_query")
        self.assertTrue(
            torch.equal(baseline.fused_logits, candidate.fused_logits)
        )
        self.assertTrue(
            torch.equal(baseline.sensor_attention, candidate.sensor_attention)
        )
        self.assertEqual(int(torch.count_nonzero(candidate.sensor_logits[invalid])), 0)

        loss_a = tq.transient_query_loss(baseline, labels)
        modified_sensor_logits = baseline.sensor_logits.clone()
        modified_sensor_logits[invalid] = 1_000_000
        synthetic = tq.TransientQueryOutput(
            fused_logits=baseline.fused_logits,
            sensor_logits=modified_sensor_logits,
            sensor_valid=baseline.sensor_valid,
            sensor_evidence=baseline.sensor_evidence,
            sensor_attention=baseline.sensor_attention,
            effective_time_valid=baseline.effective_time_valid,
        )
        loss_b = tq.transient_query_loss(synthetic, labels)
        self.assertTrue(torch.equal(loss_a.auxiliary, loss_b.auxiliary))

    def test_05_deterministic_batches_metrics_and_training_utilities(self) -> None:
        features, valid, labels = self._inputs()
        order_a = tq.fixed_epoch_batches(
            8, batch_size=3, seed=41, epoch=2, shuffle=True
        )
        order_b = tq.fixed_epoch_batches(
            8, batch_size=3, seed=41, epoch=2, shuffle=True
        )
        self.assertTrue(
            torch.equal(torch.cat(order_a), torch.cat(order_b))
        )

        models, _, _ = tq.build_matched_models(
            {
                "feature_dim": 16,
                "model_dim": 32,
                "num_heads": 4,
                "depth": 2,
                "dropout": 0.0,
            },
            seed=43,
            arms=("transient_query",),
        )
        model = models["transient_query"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        events = [f"event-{index}" for index in range(8)]
        plumes = [f"plume-{index}" for index in range(8)]
        train = tq.train_head_epoch(
            model,
            optimizer,
            features=features,
            valid_mask=valid,
            labels=labels,
            arm="transient_query",
            event_ids=events,
            plume_ids=plumes,
            batch_size=4,
            seed=47,
            epoch=1,
        )
        self.assertEqual(train["rows"], 8)
        self.assertTrue(np.isfinite(train["loss"]))

        metrics, probability = tq.evaluate_head(
            model,
            features=features,
            valid_mask=valid,
            labels=labels,
            arm="transient_query",
            batch_size=3,
        )
        self.assertEqual(probability.shape, (8,))
        for key in (
            "ap",
            "auc",
            "macro_f1_at_0_5",
            "balanced_accuracy_at_0_5",
            "best_macro_f1",
            "best_macro_f1_threshold",
        ):
            self.assertIn(key, metrics["overall"])
        self.assertEqual(metrics["sensor_count"]["single_sensor"]["rows"], 0)
        self.assertEqual(metrics["sensor_count"]["multisensor"]["rows"], 8)
        self.assertIn("s2+l89", metrics["availability"])

    def test_06_rows_without_any_t0_are_rejected(self) -> None:
        features, valid, _ = self._inputs()
        valid[0, :, 0] = False
        with self.assertRaisesRegex(ValueError, "at least one valid sensor t0"):
            self._model()(features, valid)

    def test_07_large_donor_indexing_is_deterministic(self) -> None:
        # This size makes the old O(N^2*S) implementation impractical while
        # remaining a fast structural regression for the indexed algorithm.
        rows = 5_000
        valid = torch.ones(rows, 4, 3, dtype=torch.bool)
        valid[::3, 1, 2] = False
        valid[::5, 2, 1] = False
        events = [f"event-{index}" for index in range(rows)]
        plumes = [f"plume-{index}" for index in range(rows)]
        first = tq.build_history_shuffle_donors(
            valid, events, plume_ids=plumes, seed=53
        )
        second = tq.build_history_shuffle_donors(
            valid, events, plume_ids=plumes, seed=53
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (rows, 4))
        self.assertTrue((first >= 0).all())
        sample = torch.arange(0, rows, 137)
        for row in sample.tolist():
            for sensor in range(4):
                donor = int(first[row, sensor])
                self.assertNotEqual(events[row], events[donor])
                self.assertNotEqual(plumes[row], plumes[donor])


if __name__ == "__main__":
    unittest.main(verbosity=2)
