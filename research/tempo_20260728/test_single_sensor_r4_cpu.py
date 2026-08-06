#!/usr/bin/env python3
"""CPU contracts for the bounded EMIT and approximate-S5P R4 runners."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from research.tempo_20260728.tempo_s5p_native_r4 import (
    S5PNativeR4Residual,
    shuffled_history_cache,
)
from research.tempo_20260728.tempo_single_sensor_r4 import (
    SingleSensorR4Residual,
    cross_event_history_shuffle,
)


class _DummyT0Base(nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        unique_mask: torch.Tensor,
        role_index: torch.Tensor,
    ) -> torch.Tensor:
        del valid_mask, unique_mask, role_index
        return features[:, 0].float().mean(dim=-1)


class SingleSensorR4Contracts(unittest.TestCase):
    def test_cls_zero_initialization_is_exact_p0(self) -> None:
        torch.manual_seed(3)
        features = torch.randn(5, 3, 8)
        valid = torch.ones(5, 3, dtype=torch.bool)
        unique = torch.ones_like(valid)
        roles = torch.arange(3)
        base = _DummyT0Base()
        expected = base(features, valid, unique, roles)
        model = SingleSensorR4Residual(
            base,
            feature_dim=8,
            num_roles=3,
            t0_index=0,
            model_dim=4,
            residual_cap=1.0,
        ).eval()
        for arm in ("raw_delta", "r4_motion_excitation"):
            actual, aux = model(
                features,
                valid,
                unique,
                roles,
                arm=arm,
                return_aux=True,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                aux["residual"], torch.zeros_like(expected), rtol=0, atol=0
            )

    def test_cls_shuffle_preserves_current_and_changes_history(self) -> None:
        features = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float()
        data = {
            "features": features,
            "valid_mask": torch.ones(4, 3, dtype=torch.bool),
            "unique_mask": torch.ones(4, 3, dtype=torch.bool),
            "delta_days": torch.arange(12).reshape(4, 3).float(),
            "event_ids": ["a", "b", "c", "d"],
        }
        shuffled = cross_event_history_shuffle(data, seed=17, t0_index=0)
        self.assertTrue(torch.equal(shuffled["features"][:, 0], features[:, 0]))
        self.assertFalse(
            torch.equal(shuffled["features"][:, 1:], features[:, 1:])
        )

    def test_s5p_zero_initialization_and_shuffle_contract(self) -> None:
        torch.manual_seed(5)
        features = torch.randn(4, 6, 3, 3)
        valid = torch.ones_like(features, dtype=torch.bool)
        base_logits = torch.linspace(-1.0, 1.0, 4)
        model = S5PNativeR4Residual(
            train_mean=0.0,
            train_std=1.0,
            model_dim=4,
            residual_cap=1.0,
        ).eval()
        for arm in ("raw_delta", "r4_motion_excitation"):
            actual, aux = model(
                features,
                valid,
                base_logits,
                arm=arm,
                return_aux=True,
            )
            torch.testing.assert_close(actual, base_logits, rtol=0, atol=0)
            torch.testing.assert_close(
                aux["residual"],
                torch.zeros_like(base_logits),
                rtol=0,
                atol=0,
            )
        data = {
            "features": features,
            "valid_mask": valid,
            "event_ids": ["a", "b", "c", "d"],
        }
        shuffled = shuffled_history_cache(data, seed=19)
        self.assertTrue(torch.equal(shuffled["features"][:, 0], features[:, 0]))
        self.assertFalse(
            torch.equal(shuffled["features"][:, 1:], features[:, 1:])
        )


if __name__ == "__main__":
    unittest.main()
