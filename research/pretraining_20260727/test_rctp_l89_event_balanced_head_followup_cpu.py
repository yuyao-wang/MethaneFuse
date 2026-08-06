#!/usr/bin/env python3
"""CPU regression tests for the post-hoc event-balanced RCTP head audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (
    l89_ragged_cls_experiment as base,
)
from research.pretraining_20260727 import (
    rctp_l89_event_balanced_head_followup as runner,
)


class RCTPEventBalancedHeadTests(unittest.TestCase):
    def test_each_event_has_equal_total_row_weight(self) -> None:
        event_ids = ["A", "A", "B", "C", "C", "C", "D", "D", "D", "D"]
        raw = runner.event_unit_row_weights(event_ids)
        scaled = runner.mean_one_event_weights(event_ids)
        for weights, expected in (
            (raw, 1.0),
            (scaled, len(event_ids) / len(set(event_ids))),
        ):
            totals = {}
            for event_id, weight in zip(event_ids, weights):
                totals[event_id] = totals.get(event_id, 0.0) + float(weight)
            self.assertEqual(set(totals), set(event_ids))
            self.assertTrue(
                np.allclose(
                    list(totals.values()),
                    expected,
                    atol=1e-12,
                    rtol=0.0,
                )
            )
        self.assertAlmostEqual(float(scaled.mean()), 1.0, places=12)

    def test_three_arms_load_byte_identical_initial_state(self) -> None:
        state, signature = runner.build_initial_state(
            feature_dim=16,
            num_roles=6,
            model_dim=32,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1,
            periods_days=(1.0, 3.0, 30.0),
            t0_index=0,
            seed=20260728,
        )
        expected_sha = base.state_dict_sha256(state)
        shas = {}
        signatures = {}
        for arm in runner.ARMS:
            model = runner.instantiate_from_initial_state(
                state,
                feature_dim=16,
                num_roles=6,
                model_dim=32,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.1,
                periods_days=(1.0, 3.0, 30.0),
                t0_index=0,
            )
            shas[arm] = base.state_dict_sha256(model.state_dict())
            signatures[arm] = base.model_parameter_signature(model)
        self.assertEqual(set(shas.values()), {expected_sha})
        self.assertEqual(set(shas), set(runner.ARMS))
        for arm in runner.ARMS:
            self.assertEqual(signatures[arm], signature)


if __name__ == "__main__":
    unittest.main(verbosity=2)
