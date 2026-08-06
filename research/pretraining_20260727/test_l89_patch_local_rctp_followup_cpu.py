#!/usr/bin/env python3
"""CPU contracts for the frozen final-patch RCTP follow-up."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from l89_patch_local_rctp_followup import (  # noqa: E402
    LocalLogitResidual,
    compute_local_patch_evidence,
    event_balanced_training_weights,
)


class LocalEvidenceTests(unittest.TestCase):
    def test_exact_matched_history_delta_and_missing_history(self):
        # B=2, T=3, P=2, D=4.  Identity projection makes the pooled
        # components directly inspectable.
        tokens = torch.zeros(2, 3, 2, 4)
        tokens[0, 0, 0] = torch.tensor([3.0, 0.0, 0.0, 0.0])
        tokens[0, 0, 1] = torch.tensor([0.0, 5.0, 0.0, 0.0])
        tokens[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        tokens[0, 1, 1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
        tokens[0, 2] = tokens[0, 1]
        # Row 1 has no valid history and must emit exact zero evidence.
        tokens[1, 0] = 9.0
        valid = torch.tensor(
            [[True, True, True], [True, False, False]]
        )
        evidence, count, keep = compute_local_patch_evidence(
            tokens,
            valid,
            torch.eye(4),
            t0_index=0,
            topk_fraction=0.5,
        )
        self.assertEqual(keep, 1)
        self.assertEqual(count.tolist(), [2, 0])
        # Patch 1 has larger RMS delta and is selected: signed [0,3,0,0].
        torch.testing.assert_close(
            evidence[0, :4], torch.tensor([0.0, 3.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            evidence[0, 4:8], torch.tensor([0.0, 3.0, 0.0, 0.0])
        )
        # Global signed mean is ([2,0,0,0] + [0,3,0,0]) / 2.
        torch.testing.assert_close(
            evidence[0, 8:], torch.tensor([1.0, 1.5, 0.0, 0.0])
        )
        self.assertEqual(int(torch.count_nonzero(evidence[1])), 0)

    def test_t0_is_never_accepted_as_its_own_history(self):
        tokens = torch.randn(1, 2, 3, 4)
        with self.assertRaisesRegex(ValueError, "valid unique t0"):
            compute_local_patch_evidence(
                tokens,
                torch.tensor([[False, True]]),
                torch.eye(4),
                t0_index=0,
                topk_fraction=0.5,
            )


class ResidualHeadTests(unittest.TestCase):
    def test_epoch_zero_is_bit_exact_and_has_no_bias(self):
        model = LocalLogitResidual(12, residual_cap=1.5)
        self.assertTrue(model.exact_noop)
        self.assertIsNone(model.readout.bias)
        base = torch.tensor([-3.25, 0.0, 4.5], dtype=torch.float32)
        evidence = torch.randn(3, 12)
        output = model(base, evidence)
        self.assertTrue(torch.equal(output, base))

    def test_first_step_uses_only_local_evidence(self):
        model = LocalLogitResidual(4, residual_cap=1.5)
        base = torch.tensor([0.2, -0.3])
        evidence = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
        )
        target = torch.tensor([1.0, 0.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(base, evidence), target
        )
        loss.backward()
        self.assertGreater(float(model.readout.weight.grad.abs().sum()), 0.0)
        before = model(base, evidence).detach().clone()
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        after = model(base, evidence).detach()
        self.assertGreater(float((after - before).abs().max()), 0.0)


class WeightingTests(unittest.TestCase):
    def test_event_weighting_equalizes_event_mass_and_classes(self):
        labels = torch.tensor([0, 0, 0, 1, 1])
        events = ["large", "large", "large", "p", "q"]
        weights = event_balanced_training_weights(labels, events)
        large = float(weights[:3].sum())
        positives = float(weights[3:].sum())
        self.assertAlmostEqual(large, positives, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
