#!/usr/bin/env python3
"""CPU tests for the clean primary three-seed ensemble arithmetic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.tempo_20260728 import audit_l89_clean_fixed_ensemble as target


class FixedEnsembleTests(unittest.TestCase):
    def test_mean_d1_logits_precede_p5_fusion(self) -> None:
        p5 = np.asarray([0.8, 0.2], dtype=np.float64)
        d1 = [
            np.asarray([0.9, 0.4]),
            np.asarray([0.6, 0.3]),
            np.asarray([0.7, 0.8]),
        ]
        mean_d1, candidate = target.fixed_ensemble(p5, d1)
        expected_mean_logit = np.mean(
            np.stack([target.logit(value) for value in d1]), axis=0
        )
        expected_candidate = target.sigmoid(
            0.5 * target.logit(p5) + 0.5 * expected_mean_logit
        )
        np.testing.assert_allclose(
            target.logit(mean_d1), expected_mean_logit, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            candidate, expected_candidate, rtol=0, atol=1e-12
        )

    def test_probability_average_is_not_substituted(self) -> None:
        p5 = np.asarray([0.5], dtype=np.float64)
        d1 = [
            np.asarray([0.99]),
            np.asarray([0.51]),
            np.asarray([0.51]),
        ]
        mean_d1, _ = target.fixed_ensemble(p5, d1)
        self.assertNotAlmostEqual(float(mean_d1[0]), float(np.mean(d1)), places=6)

    def test_alignment_rejects_order_change(self) -> None:
        reference = pd.DataFrame(
            {
                "id": ["1", "2"],
                "plume_id": ["p1", "p2"],
                "event_id": ["a", "b"],
                "label": [0, 1],
                "probability": [0.1, 0.9],
            }
        )
        reversed_frame = reference.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "ordered id"):
            target.aligned_frames(
                {"reference": reference, "reversed": reversed_frame}
            )

    def test_requires_three_d1_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exactly three"):
            target.fixed_ensemble(np.asarray([0.5]), [np.asarray([0.5])])

    def test_formal_bootstrap_seed_is_hard_locked(self) -> None:
        args = SimpleNamespace(
            seeds=target.FROZEN_SEEDS,
            replicates=target.FROZEN_BOOTSTRAP_REPLICATES,
            bootstrap_seed=7,
        )
        with self.assertRaisesRegex(ValueError, "seed is frozen"):
            target.run(args)

    def test_bootstrap_reports_fixed_threshold_fp_mass_and_win_rate(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        events = ["negative", "negative", "positive", "positive"]
        probabilities = {
            "candidate": np.asarray([0.1, 0.2, 0.8, 0.9]),
            "reference": np.asarray([0.6, 0.7, 0.3, 0.4]),
        }
        metrics = {
            name: target.tempo.metric_bundle(labels, values, events)
            for name, values in probabilities.items()
        }
        result = target.fixed_threshold_event_bootstrap(
            labels,
            events,
            probabilities,
            metrics,
            (("candidate_minus_reference", "candidate", "reference"),),
            replicates=20,
            seed=7,
        )
        comparison = result["candidate_minus_reference"]
        self.assertIn("all_negative_fp_mass", comparison)
        self.assertEqual(
            comparison["all_negative_fp_mass"]["better_direction"], "lower"
        )
        self.assertGreaterEqual(
            comparison["event_balanced_ap"]["win_probability"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
