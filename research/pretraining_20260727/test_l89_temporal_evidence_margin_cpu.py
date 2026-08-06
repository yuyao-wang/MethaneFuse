#!/usr/bin/env python3
"""Dataset-free CPU tests for the L8/9 temporal-evidence margin."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset import dino_classifier_head_l89_temporal_satmae as l89


class TemporalEvidenceMarginTests(unittest.TestCase):
    def test_01_hand_computed_margin_and_satisfaction(self) -> None:
        full_logits = torch.tensor(
            [[0.0, 2.0], [0.0, -1.0], [0.0, 0.5]]
        )
        current_logits = torch.tensor(
            [[0.0, 1.0], [0.0, 0.0], [0.0, 1.5]]
        )
        labels = torch.tensor([1, 0, 1])

        observed, satisfaction, signed_delta = (
            l89.label_conditioned_temporal_margin_loss(
                full_logits,
                current_logits,
                labels,
                margin=0.25,
            )
        )
        expected_signed_delta = torch.tensor([1.0, 1.0, -1.0])
        expected = F.softplus(0.25 - expected_signed_delta).mean()

        self.assertTrue(torch.equal(signed_delta, expected_signed_delta))
        self.assertTrue(torch.equal(observed, expected))
        self.assertAlmostEqual(float(satisfaction), 2.0 / 3.0, places=7)

    def test_02_total_objective_matches_three_manual_terms(self) -> None:
        full_logits = torch.tensor(
            [[0.2, 1.1], [0.7, -0.4], [-0.3, 0.5]]
        )
        current_logits = torch.tensor(
            [[0.1, 0.6], [0.2, 0.3], [-0.1, 0.7]]
        )
        labels = torch.tensor([1, 0, 1])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        observed = l89.temporal_evidence_classification_objective(
            full_logits,
            labels,
            criterion,
            enabled=True,
            current_logits=current_logits,
            margin=0.25,
            current_ce_weight=1.0,
            margin_weight=1.0,
        )
        expected_full_ce = criterion(full_logits, labels)
        expected_current_ce = criterion(current_logits, labels)
        sign = labels.float().mul(2.0).sub(1.0)
        expected_margin = F.softplus(
            0.25
            - sign
            * (
                (full_logits[:, 1] - full_logits[:, 0])
                - (
                    current_logits[:, 1] - current_logits[:, 0]
                ).detach()
            )
        ).mean()

        self.assertTrue(torch.equal(observed.full_ce, expected_full_ce))
        self.assertTrue(torch.equal(observed.current_ce, expected_current_ce))
        self.assertTrue(torch.equal(observed.margin, expected_margin))
        self.assertTrue(
            torch.equal(
                observed.total,
                expected_full_ce + expected_current_ce + expected_margin,
            )
        )

    def test_03_margin_gradient_direction_and_current_stop_gradient(self) -> None:
        full_logits = torch.zeros(2, 2, requires_grad=True)
        current_logits = torch.zeros(2, 2, requires_grad=True)
        labels = torch.tensor([1, 0])

        loss, _, _ = l89.label_conditioned_temporal_margin_loss(
            full_logits,
            current_logits,
            labels,
            margin=0.25,
        )
        loss.backward()

        self.assertLess(float(full_logits.grad[0, 1]), 0.0)
        self.assertGreater(float(full_logits.grad[1, 1]), 0.0)
        self.assertGreater(float(full_logits.grad[0, 0]), 0.0)
        self.assertLess(float(full_logits.grad[1, 0]), 0.0)
        self.assertIsNone(current_logits.grad)

    def test_04_zero_margin_weight_is_matched_aux_ce_control(self) -> None:
        full_logits = torch.tensor([[0.2, 1.1], [0.7, -0.4]])
        current_logits = torch.tensor([[0.1, 0.6], [0.2, 0.3]])
        labels = torch.tensor([1, 0])
        criterion = nn.CrossEntropyLoss()
        observed = l89.temporal_evidence_classification_objective(
            full_logits,
            labels,
            criterion,
            enabled=True,
            current_logits=current_logits,
            margin=0.25,
            current_ce_weight=1.0,
            margin_weight=0.0,
        )
        expected = criterion(full_logits, labels) + criterion(
            current_logits, labels
        )
        self.assertTrue(torch.equal(observed.total, expected))
        self.assertIsNotNone(observed.margin)

    def test_05_margin_is_invariant_to_common_logit_shifts(self) -> None:
        full_logits = torch.tensor([[0.2, 1.1], [0.7, -0.4]])
        current_logits = torch.tensor([[0.1, 0.6], [0.2, 0.3]])
        labels = torch.tensor([1, 0])
        baseline = l89.label_conditioned_temporal_margin_loss(
            full_logits,
            current_logits,
            labels,
            margin=0.25,
        )
        shifted = l89.label_conditioned_temporal_margin_loss(
            full_logits + torch.tensor([[7.0], [-3.0]]),
            current_logits + torch.tensor([[-11.0], [5.0]]),
            labels,
            margin=0.25,
        )
        for observed, expected in zip(shifted, baseline):
            self.assertTrue(torch.allclose(observed, expected, atol=1e-6))

    def test_06_current_branch_margin_gradient_is_only_its_ce(self) -> None:
        full_logits = torch.tensor(
            [[0.1, 0.2], [-0.3, 0.4]], requires_grad=True
        )
        current_logits = torch.tensor(
            [[0.5, -0.2], [0.1, 0.8]], requires_grad=True
        )
        labels = torch.tensor([1, 0])
        criterion = nn.CrossEntropyLoss()

        losses = l89.temporal_evidence_classification_objective(
            full_logits,
            labels,
            criterion,
            enabled=True,
            current_logits=current_logits,
            margin=0.25,
        )
        losses.total.backward()
        observed_current_grad = current_logits.grad.detach().clone()

        current_only = current_logits.detach().clone().requires_grad_(True)
        criterion(current_only, labels).backward()
        self.assertTrue(
            torch.equal(observed_current_grad, current_only.grad)
        )

    def test_07_flag_off_is_exact_legacy_ce(self) -> None:
        logits_expected = torch.tensor(
            [[0.4, -0.7], [-0.2, 1.3], [0.8, 0.1]],
            requires_grad=True,
        )
        logits_observed = logits_expected.detach().clone().requires_grad_(True)
        labels = torch.tensor([0, 1, 0])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        expected = criterion(logits_expected, labels)
        observed = l89.temporal_evidence_classification_objective(
            logits_observed,
            labels,
            criterion,
            enabled=False,
            current_logits=None,
            margin=float("nan"),
            current_ce_weight=-1.0,
            margin_weight=-1.0,
        )
        self.assertTrue(torch.equal(observed.total, expected))
        self.assertIs(observed.total, observed.full_ce)
        self.assertIsNone(observed.current_ce)
        self.assertIsNone(observed.margin)
        self.assertIsNone(observed.margin_satisfaction)

        expected.backward()
        observed.total.backward()
        self.assertTrue(
            torch.equal(logits_observed.grad, logits_expected.grad)
        )

        legacy_args = SimpleNamespace(
            train_csv="/tmp/train.csv",
            test_csv="/tmp/test.csv",
            train_backbone=False,
            input_mode="raw",
        )
        self.assertEqual(
            l89.default_run_name(legacy_args),
            "train__test__satmae_time__temporal_head",
        )

    @staticmethod
    def _toy_batch() -> dict:
        return {
            "imgs": torch.tensor([10.0, 20.0, 30.0]).reshape(
                1, 3, 1, 1, 1
            ),
            "chn_ids": torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1),
            "timestamps": torch.tensor(
                [[[2001.0, 1.0, 2.0], [2002.0, 3.0, 4.0], [2003.0, 5.0, 6.0]]]
            ),
            "sample_ids": ["kept"],
        }

    def test_08_raw_current_view_uses_named_t0_without_mutation(self) -> None:
        full = self._toy_batch()
        current = l89.slice_current_temporal_view(
            full,
            input_mode="raw",
            path_columns=("path_prev1", "path_t0", "path_year"),
        )

        self.assertEqual(tuple(current["imgs"].shape), (1, 1, 1, 1, 1))
        self.assertEqual(float(current["imgs"].item()), 20.0)
        self.assertEqual(float(current["chn_ids"].item()), 2.0)
        self.assertTrue(
            torch.equal(
                current["timestamps"],
                torch.tensor([[[2002.0, 3.0, 4.0]]]),
            )
        )
        self.assertIs(current["sample_ids"], full["sample_ids"])
        self.assertEqual(tuple(full["imgs"].shape), (1, 3, 1, 1, 1))

    def test_09_current_residual_keeps_first_absolute_view(self) -> None:
        full = self._toy_batch()
        current = l89.slice_current_temporal_view(
            full,
            input_mode="current_residual",
            path_columns=(
                "path_prev1",
                "path_t0",
                "path_year",
                "path_unused",
            ),
        )
        self.assertEqual(float(current["imgs"].item()), 10.0)
        self.assertEqual(float(current["chn_ids"].item()), 1.0)
        self.assertTrue(
            torch.equal(
                current["timestamps"],
                torch.tensor([[[2001.0, 1.0, 2.0]]]),
            )
        )

    def test_10_residual_and_current_modes_are_strictly_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "contains only t0-history differences"
        ):
            l89.slice_current_temporal_view(
                self._toy_batch(),
                input_mode="residual",
                path_columns=("path_t0", "path_prev1", "path_year"),
            )

        # Disabled means legacy configurations are deliberately not validated.
        l89.validate_temporal_evidence_configuration(
            enabled=False,
            input_mode="residual",
            source_timepoints=3,
            residual_slots=("path_prev1",),
            margin=float("nan"),
            current_ce_weight=-1.0,
            margin_weight=-1.0,
        )
        for unsupported_mode in ("residual", "current"):
            with self.assertRaisesRegex(ValueError, "supports only"):
                l89.validate_temporal_evidence_configuration(
                    enabled=True,
                    input_mode=unsupported_mode,
                    source_timepoints=3,
                    residual_slots=("path_prev1",),
                    margin=0.25,
                    current_ce_weight=1.0,
                    margin_weight=1.0,
                )

        for supported_mode in ("raw", "current_residual"):
            l89.validate_temporal_evidence_configuration(
                enabled=True,
                input_mode=supported_mode,
                source_timepoints=3,
                residual_slots=("path_prev1",),
                margin=0.25,
                current_ce_weight=1.0,
                margin_weight=1.0,
            )


if __name__ == "__main__":
    unittest.main()
