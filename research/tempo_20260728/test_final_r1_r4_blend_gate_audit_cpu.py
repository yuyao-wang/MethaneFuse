#!/usr/bin/env python3

import unittest

import numpy as np

from research.tempo_20260728.final_r1_r4_blend_gate_audit import (
    logit,
    sigmoid,
)


class FinalBlendAuditTest(unittest.TestCase):
    def test_equal_logit_blend_is_symmetric(self):
        first = np.asarray([0.1, 0.3, 0.8])
        second = np.asarray([0.9, 0.6, 0.2])
        left = sigmoid(0.5 * (logit(first) + logit(second)))
        right = sigmoid(0.5 * (logit(second) + logit(first)))
        np.testing.assert_allclose(left, right, atol=0.0, rtol=0.0)

    def test_equal_probabilities_are_fixed_points(self):
        value = np.asarray([0.1, 0.5, 0.9])
        blended = sigmoid(0.5 * (logit(value) + logit(value)))
        np.testing.assert_allclose(blended, value, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
