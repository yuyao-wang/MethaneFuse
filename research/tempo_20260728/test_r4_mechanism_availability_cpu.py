#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from research.tempo_20260728.audit_r4_mechanism_availability import (
    group_transition_audit,
    history_availability_category,
    shuffled_sensor_history_subset,
)


class R4MechanismAvailabilityAuditTest(unittest.TestCase):
    def test_history_availability_categories_are_exclusive(self):
        valid = np.zeros((5, 4, 3), dtype=bool)
        valid[1, 3, 0] = True
        valid[2, 3, :2] = True
        valid[3, 3, (0, 2)] = True
        valid[4, 3, :] = True
        category = history_availability_category(valid, 3)
        self.assertEqual(
            category.tolist(),
            ["absent", "t0_only", "short_only", "long_only", "both"],
        )

    def test_single_sensor_shuffle_preserves_every_other_tensor(self):
        torch.manual_seed(5)
        features = torch.randn(9, 4, 3, 6)
        valid = torch.ones(9, 4, 3, dtype=torch.bool)
        affected, shuffled = shuffled_sensor_history_subset(
            features, valid, sensor_index=1, seed=7
        )
        self.assertTrue(torch.equal(affected, torch.arange(9)))
        torch.testing.assert_close(shuffled[:, :, 0], features[:, :, 0])
        torch.testing.assert_close(shuffled[:, 0], features[:, 0])
        torch.testing.assert_close(shuffled[:, 2:], features[:, 2:])
        for role in (1, 2):
            before = torch.sort(features[:, 1, role, 0]).values
            after = torch.sort(shuffled[:, 1, role, 0]).values
            torch.testing.assert_close(before, after)

    def test_transition_counts_use_each_models_frozen_threshold(self):
        audit = group_transition_audit(
            labels=np.asarray([1, 1, 0, 0]),
            event_ids=np.asarray(["p0", "p1", "n0", "n1"]),
            p0_probability=np.asarray([0.4, 0.8, 0.6, 0.2]),
            r4_probability=np.asarray([0.7, 0.4, 0.3, 0.8]),
            p0_threshold=0.5,
            r4_threshold=0.6,
            canonical_all_negative_events={"n0", "n1"},
        )
        transition = audit["decision_transitions"]
        self.assertEqual(
            transition["false_negative_corrected_to_true_positive"], 1
        )
        self.assertEqual(
            transition["true_positive_lost_to_false_negative"], 1
        )
        self.assertEqual(
            transition["false_positive_corrected_to_true_negative"], 1
        )
        self.assertEqual(
            transition["new_false_positive_from_true_negative"], 1
        )


if __name__ == "__main__":
    unittest.main()
