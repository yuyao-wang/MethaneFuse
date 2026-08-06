#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from research.tempo_20260728.tempo_single_sensor_specialist import (
    best_branch_threshold,
    current_sensor_counts,
    evaluate_gate,
)


class SingleSensorSpecialistTest(unittest.TestCase):
    def test_current_sensor_counts_uses_only_t0(self):
        valid = torch.zeros(3, 4, 3, dtype=torch.bool)
        valid[0, 0, 0] = True
        valid[0, 1, 1] = True
        valid[1, :2, 0] = True
        valid[2, :3, 0] = True
        np.testing.assert_array_equal(
            current_sensor_counts(valid), np.asarray([1, 2, 3])
        )

    def test_branch_threshold_never_changes_other_branch(self):
        labels = np.asarray([1, 0, 1, 0, 1, 0])
        scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.1, 0.2])
        branch = np.asarray([True, True, True, True, False, False])
        fixed = np.asarray([False, False, False, False, True, False])
        threshold = best_branch_threshold(
            labels=labels,
            branch_scores=scores,
            branch_mask=branch,
            fixed_other_predictions=fixed,
        )
        prediction = np.where(branch, scores >= threshold, fixed)
        self.assertTrue(prediction[4])
        self.assertFalse(prediction[5])
        self.assertAlmostEqual(threshold, 0.7)

    def test_gate_optimization_can_outperform_fixed_threshold(self):
        labels = np.asarray([1, 0, 1, 0])
        specialist = np.asarray([0.6, 0.4, 0.2, 0.1])
        p0 = np.asarray([0.1, 0.1, 0.8, 0.2])
        counts = np.asarray([1, 1, 2, 2])
        metrics, score, prediction, threshold = evaluate_gate(
            labels=labels,
            specialist_scores=specialist,
            p0_scores=p0,
            sensor_counts=counts,
            p0_threshold=0.5,
            specialist_threshold=None,
        )
        self.assertEqual(metrics["binary_f1"], 1.0)
        self.assertEqual(threshold, 0.6)
        np.testing.assert_array_equal(
            score, np.asarray([0.6, 0.4, 0.8, 0.2])
        )
        np.testing.assert_array_equal(
            prediction, np.asarray([True, False, True, False])
        )


if __name__ == "__main__":
    unittest.main()
