#!/usr/bin/env python3

import unittest

import numpy as np

from research.tempo_20260728.availability_gate_control import fixed_metrics


class TempoLockedEvaluatorTest(unittest.TestCase):
    def test_availability_gate_uses_branch_specific_locked_predictions(self):
        sensor_count = np.asarray([1, 2, 1, 3])
        p0_score = np.asarray([0.2, 0.7, 0.8, 0.4])
        r4_score = np.asarray([0.6, 0.1, 0.3, 0.9])
        p0_prediction = p0_score >= 0.5
        r4_prediction = r4_score >= 0.4
        use_r4 = sensor_count == 1
        gate_score = np.where(use_r4, r4_score, p0_score)
        gate_prediction = np.where(
            use_r4, r4_prediction, p0_prediction
        )
        np.testing.assert_array_equal(
            gate_score, np.asarray([0.6, 0.7, 0.3, 0.4])
        )
        np.testing.assert_array_equal(
            gate_prediction, np.asarray([True, True, False, False])
        )

    def test_fixed_metrics_never_optimizes_the_supplied_prediction(self):
        labels = np.asarray([1, 1, 0, 0])
        score = np.asarray([0.1, 0.2, 0.9, 0.8])
        supplied = np.asarray([True, True, False, False])
        metrics = fixed_metrics(labels, score, supplied)
        self.assertEqual(metrics["binary_f1"], 1.0)
        self.assertLess(metrics["ap"], 0.5)


if __name__ == "__main__":
    unittest.main()
