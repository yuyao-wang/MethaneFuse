#!/usr/bin/env python3
"""CPU contracts for the fixed-bin L89 TEMPO mechanism audit."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.tempo_20260728 import tempo_l89_mechanism_audit as audit


class MechanismAuditTests(unittest.TestCase):
    def test_semantic_lag_bins_are_fixed(self) -> None:
        self.assertEqual(
            audit.semantic_lag_bin(float("nan"), nearest=True),
            "00_no_unique_history",
        )
        self.assertEqual(
            audit.semantic_lag_bin(7.0, nearest=True), "01_le_7d"
        )
        self.assertEqual(
            audit.semantic_lag_bin(17.0, nearest=True), "03_17_24d"
        )
        self.assertEqual(
            audit.semantic_lag_bin(365.0, nearest=False), "02_351_365d"
        )
        self.assertEqual(
            audit.semantic_lag_bin(365.1, nearest=False), "03_gt_365d"
        )

    def test_single_class_bin_has_null_ranking_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "event_id": ["a", "a", "b"],
                "event_kind": [
                    "all_negative_event",
                    "all_negative_event",
                    "all_negative_event",
                ],
                "label": [0, 0, 0],
            }
        )
        metrics = audit.fixed_threshold_metrics(
            frame, np.asarray([0.8, 0.2, 0.1]), threshold=0.5
        )
        self.assertIsNone(metrics["event_balanced_ap"])
        self.assertIsNone(metrics["event_balanced_auc"])
        self.assertEqual(metrics["all_negative_events_with_any_fp"], 1)
        self.assertEqual(metrics["negative_row_fp_count"], 1)

    def test_transition_rates_separate_corrections_and_introductions(self) -> None:
        frame = pd.DataFrame(
            {
                "event_id": ["a", "a", "b", "b"],
                "label": [0, 1, 0, 1],
                "group": ["x", "x", "y", "y"],
            }
        )
        baseline_correct = np.asarray([False, True, True, True])
        candidate_correct = np.asarray([True, False, True, True])
        result = audit.row_transition_rates_by_group(
            frame,
            group_columns=("group",),
            baseline_correct=baseline_correct,
            candidate_correct=candidate_correct,
        )["group"]
        x = result[0]
        self.assertEqual(x["corrected_errors"], 1)
        self.assertEqual(x["introduced_errors"], 1)
        self.assertEqual(x["net_correct_rows"], 0)
        self.assertEqual(x["corrected_false_positives"], 1)
        self.assertEqual(x["introduced_false_negatives"], 1)


if __name__ == "__main__":
    unittest.main()
