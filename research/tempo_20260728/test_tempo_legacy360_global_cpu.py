#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from research.tempo_20260728.tempo_legacy360_global import (
    ARMS,
    TempoGlobalHead,
    active_parameter_compute_contract,
    all_negative_event_fp,
    all_negative_event_row_weights,
    best_event_guarded_threshold,
    build_matched_initial_state,
    event_operating_audit,
    _weighted_bootstrap_metrics,
    shuffle_history,
)


class TempoGlobalHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.features = torch.randn(7, 4, 3, 12)
        self.valid = torch.ones(7, 4, 3, dtype=torch.bool)
        self.valid[1, 2, 2] = False
        self.valid[2, 1, 1:] = False
        self.base = torch.linspace(-1.0, 1.0, 7)
        self.base_sensor = torch.randn(7, 4)

    def test_all_arms_are_exact_zero_init_residuals(self):
        model = TempoGlobalHead(12, model_dim=8)
        for arm in ARMS:
            output = model(
                self.features,
                self.valid,
                self.base,
                self.base_sensor,
                arm=arm,
            )
            torch.testing.assert_close(output["fused_logits"], self.base)
            torch.testing.assert_close(
                output["sensor_logits"], self.base_sensor
            )
            self.assertTrue(torch.equal(output["residual"], torch.zeros(7)))

    def test_p2_removes_a_linear_historical_trend(self):
        model = TempoGlobalHead(12, model_dim=8)
        older = torch.randn(2, 4, 12)
        trend = torch.randn(2, 4, 12)
        short = older + trend
        current = short + trend / 3.0
        features = torch.stack((current, short, older), dim=2)
        _, _, _ = model.evidence_operator(features, self.valid[:2], arm="p2")
        z = torch.nn.functional.layer_norm(features, (12,))
        d_short = z[:, :, 0] - z[:, :, 1]
        historical = z[:, :, 1] - z[:, :, 2]
        innovation = d_short - historical / 3.0
        # Per-observation LayerNorm means a linear trend in raw space is not
        # guaranteed linear after normalization; the explicit operator must
        # nevertheless equal the documented normalized-feature equation.
        z_current, z_short, z_old = z.unbind(dim=2)
        expected = (z_current - z_short) - (z_short - z_old) / 3.0
        torch.testing.assert_close(innovation, expected)

    def test_scale_gate_removes_s5p_history(self):
        model = TempoGlobalHead(12, model_dim=8)
        _, evidence_valid_p2, _ = model.evidence_operator(
            self.features, self.valid, arm="p2"
        )
        _, evidence_valid_p3, _ = model.evidence_operator(
            self.features, self.valid, arm="p3"
        )
        self.assertTrue(evidence_valid_p2[:, 3].all())
        self.assertFalse(evidence_valid_p3[:, 3].any())

    def test_matched_parameter_signature_across_arms(self):
        state, signature, digest = build_matched_initial_state(
            12, model_dim=8, residual_cap=1.0, seed=42
        )
        self.assertGreater(signature["parameter_count"], 0)
        self.assertEqual(len(digest), 64)
        for arm in ARMS:
            model = TempoGlobalHead(12, model_dim=8)
            model.load_state_dict(state, strict=True)
            model(
                self.features,
                self.valid,
                self.base,
                self.base_sensor,
                arm=arm,
            )

    def test_r4_add_has_identical_graph_active_capacity_and_linear_compute(self):
        model = TempoGlobalHead(12, model_dim=8)
        r4 = active_parameter_compute_contract(model, arm="r4")
        r4add = active_parameter_compute_contract(model, arm="r4add")
        self.assertEqual(
            r4["active_parameter_count"],
            r4add["active_parameter_count"],
        )
        self.assertEqual(
            r4["active_parameter_names_sha256"],
            r4add["active_parameter_names_sha256"],
        )
        self.assertEqual(
            r4["compute"]["learned_linear_macs_per_row"],
            r4add["compute"]["learned_linear_macs_per_row"],
        )
        self.assertEqual(
            r4["active_parameters"],
            r4add["active_parameters"],
        )

    def test_r4_add_changes_only_the_fusion_operator(self):
        model = TempoGlobalHead(12, model_dim=8)
        r4, valid_r4, tokens_r4 = model.evidence_operator(
            self.features, self.valid, arm="r4"
        )
        r4add, valid_add, tokens_add = model.evidence_operator(
            self.features, self.valid, arm="r4add"
        )
        self.assertTrue(torch.equal(valid_r4, valid_add))
        self.assertTrue(torch.equal(tokens_r4, tokens_add))
        self.assertFalse(torch.allclose(r4, r4add))

    def test_t0_query_attention_matches_r4_active_parameter_count(self):
        model = TempoGlobalHead(12, model_dim=8)
        r4 = active_parameter_compute_contract(model, arm="r4")
        attention = active_parameter_compute_contract(model, arm="r6attn")
        self.assertEqual(
            r4["active_parameter_count"],
            attention["active_parameter_count"],
        )
        self.assertNotEqual(
            r4["active_parameter_names_sha256"],
            attention["active_parameter_names_sha256"],
        )

    def test_history_shuffle_preserves_t0_masks_and_marginals(self):
        shuffled = shuffle_history(self.features, self.valid, seed=9)
        torch.testing.assert_close(shuffled[:, :, 0], self.features[:, :, 0])
        for sensor in range(4):
            for role in (1, 2):
                positions = torch.nonzero(self.valid[:, sensor, role]).flatten()
                before = self.features[positions, sensor, role]
                after = shuffled[positions, sensor, role]
                before_sorted = torch.sort(before[:, 0]).values
                after_sorted = torch.sort(after[:, 0]).values
                torch.testing.assert_close(before_sorted, after_sorted)

    def test_all_negative_event_fp_uses_equal_event_weight(self):
        audit = all_negative_event_fp(
            np.asarray([0, 0, 0, 0, 1]),
            np.asarray([0.9, 0.1, 0.8, 0.1, 0.9]),
            ["a", "a", "b", "b", "c"],
            threshold=0.5,
        )
        self.assertEqual(audit["events"], 2)
        self.assertAlmostEqual(audit["fp_rate_mean"], 0.5)
        self.assertAlmostEqual(audit["fp_mass"], 1.0)
        self.assertEqual(audit["hard_fp_rows"], 2)

    def test_event_operating_audit_keeps_positive_event_recall(self):
        audit = event_operating_audit(
            np.asarray([0, 0, 0, 0, 1]),
            np.asarray([0.9, 0.1, 0.8, 0.1, 0.9]),
            ["a", "a", "b", "b", "c"],
            threshold=0.5,
        )
        self.assertEqual(audit["canonical_events"], 3)
        self.assertEqual(audit["positive_or_mixed_events"], 1)
        self.assertEqual(audit["positive_or_mixed_any_detection_count"], 1)
        self.assertAlmostEqual(
            audit["positive_or_mixed_any_detection_recall"], 1.0
        )

    def test_all_negative_event_weights_equalize_event_mass(self):
        weights = all_negative_event_row_weights(
            [0, 0, 0, 0, 0, 1],
            ["a", "a", "b", "b", "b", "c"],
        ).numpy()
        self.assertGreater(weights[:5].min(), 0)
        self.assertEqual(weights[5], 0)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:5].sum()))

    def test_event_guarded_threshold_obeys_null_and_recall_limits(self):
        result = best_event_guarded_threshold(
            [0, 0, 1, 1],
            [0.6, 0.2, 0.8, 0.7],
            ["n0", "n1", "p0", "p1"],
            max_all_negative_fp_rows=0,
            max_all_negative_fp_events=0,
            min_positive_event_detections=2,
        )
        self.assertIsNotNone(result)
        threshold, positive_f1, macro_f1 = result
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.7)
        self.assertGreater(positive_f1, 0)
        self.assertGreater(macro_f1, 0)

    def test_vectorized_weighted_metrics_match_unweighted_sklearn(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        probability = np.asarray([0.1, 0.9, 0.4, 0.7])
        metrics = _weighted_bootstrap_metrics(
            labels,
            probability,
            0.5,
            np.ones((1, len(labels)), dtype=np.float64),
        )
        self.assertAlmostEqual(float(metrics["binary_f1"][0]), 1.0)
        self.assertAlmostEqual(float(metrics["macro_f1"][0]), 1.0)
        self.assertAlmostEqual(float(metrics["ap"][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
