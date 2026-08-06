#!/usr/bin/env python3

import unittest

import numpy as np

from research.tempo_20260728.availability_gate_control import (
    fixed_metrics,
    paired_event_bootstrap,
)


class AvailabilityGateControlTest(unittest.TestCase):
    def test_fixed_metrics_accepts_branch_specific_predictions(self):
        metrics = fixed_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.8, 0.1, 0.9, 0.2]),
            np.asarray([False, False, True, True]),
        )
        self.assertEqual(metrics["binary_f1"], 1.0)
        self.assertEqual(metrics["false_positive_rows"], 0)

    def test_paired_bootstrap_zero_for_identical_candidate(self):
        labels = np.asarray([0, 1, 0, 1])
        events = np.asarray(["a", "b", "c", "d"])
        score = np.asarray([0.1, 0.8, 0.2, 0.9])
        prediction = score >= 0.5
        result = paired_event_bootstrap(
            labels=labels,
            event_ids=events,
            models={
                "same": (score, prediction),
                "reference": (score.copy(), prediction.copy()),
            },
            candidate_name="same",
            references=["reference"],
            repeats=20,
            seed=4,
            batch_size=5,
        )
        for metric in ("binary_f1", "macro_f1", "ap"):
            value = result["candidate_minus_reference"]["reference"][metric]
            self.assertEqual(value["mean_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
