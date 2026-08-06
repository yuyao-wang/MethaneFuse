#!/usr/bin/env python3
"""CPU contracts for the canonical-event OOF reliability liaison."""

from __future__ import annotations

import unittest

import numpy as np

from research.tempo_20260728 import (
    tempo_l89_oof_reliability_liaison as liaison,
)


class ReliabilityLiaisonTests(unittest.TestCase):
    def test_nominal_reliability_is_one_and_degrades(self) -> None:
        nominal = liaison.reliability_score(
            np.asarray([5.0]),
            np.asarray([16.0]),
            np.asarray([365.0]),
            np.asarray([1.0]),
        )
        degraded = liaison.reliability_score(
            np.asarray([4.0]),
            np.asarray([40.0]),
            np.asarray([330.0]),
            np.asarray([0.8]),
        )
        self.assertAlmostEqual(float(nominal[0]), 1.0)
        self.assertLess(float(degraded[0]), float(nominal[0]))

    def test_fit_weights_balance_event_weighted_row_classes(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int64)
        events = ["a", "a", "b", "c", "c", "c"]
        weights, audit = liaison.event_class_balanced_fit_weights(
            labels, events
        )
        self.assertAlmostEqual(
            float(weights[labels == 0].sum()),
            float(weights[labels == 1].sum()),
            places=10,
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=10)
        self.assertAlmostEqual(
            audit["final_positive_weight_sum"],
            audit["final_negative_weight_sum"],
            places=10,
        )

    def test_oof_decisions_use_per_row_fold_threshold(self) -> None:
        probability = np.asarray([0.2, 0.6, 0.7])
        threshold = np.asarray([0.1, 0.65, 0.7])
        observed = liaison.decisions_from_fold_threshold(
            probability, threshold
        )
        np.testing.assert_array_equal(
            observed, np.asarray([True, False, True])
        )


if __name__ == "__main__":
    unittest.main()
