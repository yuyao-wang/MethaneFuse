#!/usr/bin/env python3
"""CPU-only contracts for the development-only L89 TEMPO pilot."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from research.tempo_20260728 import tempo_l89_global as tempo


class DummyFullBase(nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        unique_mask: torch.Tensor,
        delta_days: torch.Tensor,
        role_index: torch.Tensor,
    ) -> torch.Tensor:
        del unique_mask, delta_days, role_index
        return features[:, 0, 0] - 0.25 * features[:, 0, 1]


def build_model() -> tempo.TEMPOGlobalResidual:
    torch.manual_seed(7)
    return tempo.TEMPOGlobalResidual(
        DummyFullBase(),
        feature_dim=8,
        num_roles=4,
        t0_index=0,
        temporal_dim=6,
        dropout=0.0,
        periods_days=(1, 30, 365),
    ).eval()


class TempoGlobalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.features = torch.randn(5, 4, 8)
        self.unique = torch.ones(5, 4, dtype=torch.bool)
        self.delta = torch.tensor(
            [[0.0, -2.0, -30.0, -365.0]] * 5, dtype=torch.float32
        )
        self.quality = torch.ones(5, 4)
        self.roles = torch.arange(4)

    def test_zero_init_and_p0_preserve_exact_base(self) -> None:
        model = build_model()
        expected = DummyFullBase()(
            self.features, self.unique, self.delta, self.roles
        )
        for arm in (
            value
            for value in tempo.ARM_NAMES
            if value != tempo.SparseSlowFastResidual.ARM
        ):
            actual = model(
                self.features,
                self.unique,
                self.delta,
                self.quality,
                self.roles,
                arm=arm,
            )
            self.assertTrue(torch.equal(actual, expected), arm)

        with torch.no_grad():
            model.residual[-1].weight.fill_(0.5)
        p0 = model(
            self.features,
            self.unique,
            self.delta,
            self.quality,
            self.roles,
            arm="p0_base",
        )
        self.assertTrue(torch.equal(p0, expected))

    def test_d6_has_exact_zero_base_and_uses_gated_delta(self) -> None:
        model = tempo.TEMPOGlobalResidual(
            tempo.ZeroLogitBase(),
            feature_dim=8,
            num_roles=4,
            t0_index=0,
            temporal_dim=6,
            dropout=0.0,
            periods_days=(1, 30, 365),
        ).eval()
        initial, aux = model(
            self.features,
            self.unique,
            self.delta,
            self.quality,
            self.roles,
            arm="d6_onset_only",
            return_aux=True,
        )
        self.assertEqual(int(torch.count_nonzero(aux["base_logit"])), 0)
        self.assertEqual(int(torch.count_nonzero(initial)), 0)
        self.assertTrue(
            torch.allclose(
                aux["history_weights"].sum(dim=1),
                torch.ones(len(self.features)),
            )
        )
        with torch.no_grad():
            model.residual[-1].weight.fill_(0.5)
        trained = model(
            self.features,
            self.unique,
            self.delta,
            self.quality,
            self.roles,
            arm="d6_onset_only",
        )
        self.assertGreater(float(trained.abs().sum()), 0.0)

    def test_identical_sequence_has_zero_delta_and_onset(self) -> None:
        model = build_model()
        repeated = self.features[:, :1].expand(-1, 4, -1).clone()
        _, aux = model(
            repeated,
            self.unique,
            self.delta,
            self.quality,
            self.roles,
            arm="p2_onset",
            return_aux=True,
        )
        self.assertEqual(int(torch.count_nonzero(aux["current_difference"])), 0)
        self.assertEqual(int(torch.count_nonzero(aux["normal_variation"])), 0)
        self.assertEqual(int(torch.count_nonzero(aux["onset"])), 0)

    def test_gate_masks_invalid_history_and_normalizes(self) -> None:
        model = build_model()
        unique = self.unique.clone()
        unique[0, 2:] = False
        unique[1, 1:] = False
        _, aux = model(
            self.features,
            unique,
            self.delta,
            self.quality,
            self.roles,
            arm="p3_gated_onset",
            return_aux=True,
        )
        weights = aux["history_weights"]
        self.assertAlmostEqual(float(weights[0].sum()), 1.0, places=6)
        self.assertEqual(float(weights[0, 1:].sum()), 0.0)
        # No history: aggregation is suppressed after the softmax fallback.
        output, base_aux = model(
            self.features,
            unique,
            self.delta,
            self.quality,
            self.roles,
            arm="p0_base",
            return_aux=True,
        )
        self.assertEqual(float(base_aux["residual_logit"][1]), 0.0)
        self.assertTrue(torch.isfinite(output).all())

    def test_sparse_slowfast_zero_init_masks_and_independent_gates(self) -> None:
        torch.manual_seed(23)
        features = torch.randn(4, 6, 8)
        unique = torch.ones(4, 6, dtype=torch.bool)
        delta = torch.tensor(
            [[0.0, -8.0, -16.0, -24.0, -90.0, -365.0]] * 4
        )
        quality = torch.ones(4, 6)
        roles = torch.arange(6)
        model = tempo.SparseSlowFastResidual(
            DummyFullBase(),
            feature_dim=8,
            num_roles=6,
            t0_index=0,
            temporal_dim=6,
            dropout=0.0,
            periods_days=(1, 30, 365),
        ).eval()
        expected = DummyFullBase()(features, unique, delta, roles)
        initial, aux = model(
            features,
            unique,
            delta,
            quality,
            roles,
            arm=tempo.SparseSlowFastResidual.ARM,
            return_aux=True,
        )
        self.assertTrue(torch.equal(initial, expected))
        self.assertFalse(
            model.fast_gate[0].weight.data_ptr()
            == model.slow_gate[0].weight.data_ptr()
        )
        self.assertTrue(
            torch.allclose(
                aux["fast_weights"].sum(dim=1), torch.ones(len(features))
            )
        )
        self.assertTrue(
            torch.allclose(
                aux["slow_weights"].sum(dim=1), torch.ones(len(features))
            )
        )
        self.assertTrue(torch.equal(aux["fixed_branch_count"], torch.full((4,), 2.0)))

        fast_only = unique.clone()
        fast_only[:, 4:] = False
        _, fast_aux = model(
            features,
            fast_only,
            delta,
            quality,
            roles,
            arm=tempo.SparseSlowFastResidual.ARM,
            return_aux=True,
        )
        self.assertFalse(fast_aux["slow_available"].any())
        self.assertTrue(fast_aux["fast_available"].all())
        self.assertEqual(
            int(torch.count_nonzero(fast_aux["slow_weights"])), 0
        )
        self.assertTrue(
            torch.equal(
                fast_aux["fixed_branch_count"], torch.ones(len(features))
            )
        )

    def test_sparse_slowfast_is_capacity_matched_to_d1(self) -> None:
        d1 = tempo.TEMPOGlobalResidual(
            DummyFullBase(),
            feature_dim=768,
            num_roles=6,
            t0_index=0,
            temporal_dim=192,
            dropout=0.15,
            periods_days=(1, 3, 7, 30, 90, 365),
        )
        slowfast = tempo.SparseSlowFastResidual(
            DummyFullBase(),
            feature_dim=768,
            num_roles=6,
            t0_index=0,
            temporal_dim=174,
            dropout=0.15,
            periods_days=(1, 3, 7, 30, 90, 365),
        )
        d1_count = tempo.trainable_parameter_signature(d1)["parameter_count"]
        slowfast_count = tempo.trainable_parameter_signature(slowfast)[
            "parameter_count"
        ]
        self.assertEqual(d1_count, 412801)
        self.assertEqual(slowfast_count, 413744)
        self.assertLess(abs(slowfast_count / d1_count - 1.0), 0.003)

    def test_all_negative_topk_is_equal_event_and_differentiable(self) -> None:
        logits = torch.tensor([3.0, 1.0, -1.0, 2.0, -2.0], requires_grad=True)
        codes = torch.tensor([0, 0, 0, 1, 2])
        null = torch.tensor([True, True, False])
        loss = tempo.all_negative_topk_loss(
            logits, codes, null, top_k=2, margin=0.0
        )
        expected = (
            torch.nn.functional.softplus(torch.tensor([3.0, 1.0])).mean()
            + torch.nn.functional.softplus(torch.tensor([2.0])).mean()
        ) / 2.0
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        loss.backward()
        self.assertGreater(float(logits.grad[:4].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[4]), 0.0)

    def test_event_selectivity_metrics(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        scores = np.asarray([0.9, 0.1, 0.2, 0.8, 0.1, 0.7])
        events = ["n0", "n0", "n1", "p0", "p0", "p1"]
        metrics = tempo.metric_bundle(
            labels, scores, events, threshold=0.5
        )
        self.assertEqual(metrics["all_negative_event_count"], 2)
        self.assertEqual(metrics["all_negative_events_with_any_fp"], 1)
        self.assertEqual(metrics["all_negative_fp_rows"], 1)
        self.assertAlmostEqual(metrics["all_negative_event_any_fp_rate"], 0.5)
        self.assertEqual(metrics["positive_event_count"], 2)
        self.assertAlmostEqual(
            metrics["positive_event_any_detection_recall"], 1.0
        )

    def test_shuffle_preserves_t0_and_uses_cross_event_history(self) -> None:
        data = {
            "features": self.features.clone(),
            "labels": torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32),
            "valid_mask": self.unique.clone(),
            "unique_mask": self.unique.clone(),
            "delta_days": self.delta.clone(),
            "valid_fraction": self.quality.clone(),
            "event_ids": ["a", "b", "c", "d", "e"],
            "ids": [str(i) for i in range(5)],
            "plume_ids": [str(i) for i in range(5)],
        }
        shuffled, audit = tempo.shuffled_development_view(
            data, seed=19, t0_index=0, return_audit=True
        )
        self.assertTrue(torch.equal(shuffled["features"][:, 0], data["features"][:, 0]))
        self.assertFalse(
            torch.equal(shuffled["features"][:, 1:], data["features"][:, 1:])
        )
        self.assertEqual(shuffled["ids"], data["ids"])
        self.assertEqual(shuffled["plume_ids"], data["plume_ids"])
        self.assertEqual(shuffled["event_ids"], data["event_ids"])
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["availability_mismatch_rows"], 0)
        self.assertEqual(audit["availability_pattern_mismatch_count"], 0)
        self.assertTrue(audit["all_target_donor_strata_equal"])
        self.assertTrue(audit["all_cross_event"])
        self.assertEqual(audit["failed_strata"], [])
        self.assertEqual(audit["strata_count"], 1)
        self.assertEqual(audit["pattern_count"], 1)
        self.assertTrue(audit["all_donors_from_different_canonical_event"])
        for key in (
            "labels",
            "valid_mask",
            "unique_mask",
            "delta_days",
            "valid_fraction",
        ):
            self.assertTrue(torch.equal(shuffled[key], data[key]))

    def test_shuffle_donors_match_full_history_availability_strata(self) -> None:
        event_ids = ["a", "b", "c", "d", "e", "f"]
        valid = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 1],
            ],
            dtype=torch.bool,
        )
        unique = torch.tensor(
            [
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
            ],
            dtype=torch.bool,
        )
        donors, audit = tempo.build_availability_stratified_history_donors(
            event_ids,
            valid,
            unique,
            seed=31,
            t0_index=0,
        )
        donors_replayed, audit_replayed = (
            tempo.build_availability_stratified_history_donors(
                event_ids,
                valid,
                unique,
                seed=31,
                t0_index=0,
            )
        )
        self.assertTrue(torch.equal(donors, donors_replayed))
        self.assertEqual(
            audit["donor_indices_sha256"],
            audit_replayed["donor_indices_sha256"],
        )
        self.assertEqual(audit["strata_count"], 2)
        self.assertEqual(audit["availability_mismatch_rows"], 0)
        self.assertTrue(audit["all_target_donor_strata_equal"])
        self.assertTrue(audit["all_cross_event"])
        history = [1, 2, 3, 4, 5]
        self.assertTrue(
            torch.equal(valid[:, history], valid[donors][:, history])
        )
        self.assertTrue(
            torch.equal(unique[:, history], unique[donors][:, history])
        )
        for target_index, donor_index in enumerate(donors.tolist()):
            self.assertNotEqual(
                event_ids[target_index], event_ids[donor_index]
            )

    def test_shuffle_fails_closed_for_single_event_availability_stratum(
        self,
    ) -> None:
        valid = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 0],
            ],
            dtype=torch.bool,
        )
        unique = valid.clone()
        with self.assertRaisesRegex(
            ValueError, "fewer than two canonical events"
        ):
            tempo.build_availability_stratified_history_donors(
                ["only", "only", "b", "c"],
                valid,
                unique,
                seed=41,
                t0_index=0,
            )

    def test_forbidden_development_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            safe = Path(directory) / "development" / "cache.pt"
            tempo.assert_development_path(safe, purpose="unit test")
            for name in ("test", "sealed-test", "outer_holdout"):
                with self.assertRaises(ValueError):
                    tempo.assert_development_path(
                        Path(directory) / name / "cache.pt", purpose="unit test"
                    )

    def test_paired_event_bootstrap_detects_identical_and_better_scores(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
        events = ["a", "a", "b", "b", "c", "c"]
        base = np.asarray([0.4, 0.3, 0.6, 0.55, 0.45, 0.52])
        better = np.asarray([0.1, 0.2, 0.9, 0.8, 0.2, 0.75])
        probabilities = {"base": base, "copy": base.copy(), "better": better}
        metrics = {
            name: tempo.metric_bundle(labels, value, events)
            for name, value in probabilities.items()
        }
        result = tempo.paired_event_bootstrap_deltas(
            labels,
            events,
            probabilities,
            metrics,
            (
                ("copy_minus_base", "copy", "base"),
                ("better_minus_base", "better", "base"),
            ),
            replicates=40,
            seed=17,
        )
        for metric in (
            "event_balanced_ap",
            "event_balanced_auc",
            "event_balanced_positive_f1_selected",
            "event_balanced_macro_f1_selected",
        ):
            interval = result["copy_minus_base"][metric]
            self.assertEqual(interval["point"], 0.0)
            self.assertEqual(interval["ci_95_low"], 0.0)
            self.assertEqual(interval["ci_95_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
