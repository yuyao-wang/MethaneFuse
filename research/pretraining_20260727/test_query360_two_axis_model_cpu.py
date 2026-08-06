#!/usr/bin/env python3
"""CPU checks for the factorized 360 m time-by-sensor readout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from query360_model import transient_query_loss  # noqa: E402
from query360_two_axis_model import TwoAxisQuery360Head  # noqa: E402


class TwoAxisQuery360HeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.features = torch.randn(5, 4, 3, 16)
        self.valid = torch.tensor(
            [
                [[1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1]],
                [[1, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0]],
            ],
            dtype=torch.bool,
        )
        self.model = TwoAxisQuery360Head(
            16,
            model_dim=32,
            num_heads=4,
            temporal_depth=1,
            dropout=0.0,
        )

    def test_output_and_loss_contract(self) -> None:
        output = self.model(self.features, self.valid)
        self.assertEqual(tuple(output.fused_logits.shape), (5,))
        self.assertEqual(tuple(output.sensor_logits.shape), (5, 4))
        self.assertEqual(tuple(output.sensor_attention.shape), (5, 4))
        self.assertEqual(tuple(output.axis_gate.shape), (5, 32))
        self.assertTrue(torch.isfinite(output.fused_logits).all())
        self.assertTrue(
            torch.allclose(
                output.sensor_attention.sum(dim=1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                output.sensor_attention.masked_select(~output.sensor_valid),
                torch.zeros_like(
                    output.sensor_attention.masked_select(~output.sensor_valid)
                ),
            )
        )
        loss = transient_query_loss(
            output, torch.tensor([0, 1, 0, 1, 1]), auxiliary_weight=0.3
        )
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertIsNotNone(self.model.input_projection.weight.grad)

    def test_current_only_is_invariant_to_history_values(self) -> None:
        self.model.eval()
        changed = self.features.clone()
        changed[:, :, 1:] = torch.randn_like(changed[:, :, 1:]) * 1000.0
        with torch.no_grad():
            first = self.model(
                self.features, self.valid, arm="current_only"
            ).fused_logits
            second = self.model(
                changed, self.valid, arm="current_only"
            ).fused_logits
        self.assertTrue(torch.equal(first, second))

    def test_scale_aware_masks_only_s5p_history(self) -> None:
        output = self.model(
            self.features, self.valid, arm="scale_aware_two_axis_query"
        )
        self.assertFalse(output.effective_time_valid[:, 3, 1:].any())
        self.assertTrue(
            torch.equal(
                output.effective_time_valid[:, :3],
                self.valid[:, :3],
            )
        )

    def test_missing_tokens_cannot_change_logits(self) -> None:
        self.model.eval()
        changed = self.features.clone()
        changed[~self.valid] = 1e6
        with torch.no_grad():
            first = self.model(self.features, self.valid).fused_logits
            second = self.model(changed, self.valid).fused_logits
        self.assertTrue(torch.equal(first, second))

    def test_requires_a_current_sensor(self) -> None:
        bad = self.valid.clone()
        bad[0, :, 0] = False
        with self.assertRaisesRegex(ValueError, "current sensor"):
            self.model(self.features, bad)

    def test_concat_checkpoint_base_is_exact_at_initialization(self) -> None:
        self.model.eval()
        base_fused = torch.linspace(-2.0, 2.0, 5)
        base_sensor = torch.randn(5, 4)
        with torch.no_grad():
            output = self.model(
                self.features,
                self.valid,
                base_fused_logits=base_fused,
                base_sensor_logits=base_sensor,
            )
        self.assertTrue(torch.equal(output.fused_logits, base_fused))
        self.assertTrue(
            torch.equal(
                output.sensor_logits.masked_select(output.sensor_valid),
                base_sensor.masked_select(output.sensor_valid),
            )
        )


if __name__ == "__main__":
    unittest.main()
