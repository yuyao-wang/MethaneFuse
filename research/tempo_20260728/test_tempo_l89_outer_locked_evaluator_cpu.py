#!/usr/bin/env python3
"""CPU contracts for the L89 exactly-once outer evaluator."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.tempo_20260728 import tempo_l89_outer_locked_evaluator as target


class L89OuterLockedEvaluatorContracts(unittest.TestCase):
    def test_fixed_logit_fusion_is_exact_and_order_independent(self) -> None:
        p5 = np.asarray([-2.0, 0.0, 2.0])
        d1 = np.asarray([2.0, 1.0, -2.0])
        expected = np.asarray([0.0, 0.5, 0.0])
        first = target.fuse_locked_logits(
            {"p5": p5, "d1": d1}, {"p5": 0.5, "d1": 0.5}
        )
        second = target.fuse_locked_logits(
            {"d1": d1, "p5": p5}, {"d1": 0.5, "p5": 0.5}
        )
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)

    def test_fusion_rejects_unlocked_or_nonunit_weights(self) -> None:
        logits = {"p5": np.zeros(2), "d1": np.zeros(2)}
        with self.assertRaises(ValueError):
            target.fuse_locked_logits(
                logits, {"p5": 0.5, "d1": 0.4}
            )
        with self.assertRaises(ValueError):
            target.fuse_locked_logits(
                logits, {"p5": 0.5, "d1": 0.4, "patch": 0.1}
            )

    def test_cache_alignment_binds_identity_and_first_feature_block(self) -> None:
        rows = 3
        base_features = torch.randn(rows, 6, 768)
        common = {
            "labels": torch.tensor([0.0, 1.0, 0.0]),
            "unique_mask": torch.ones(rows, 6, dtype=torch.bool),
            "delta_days": torch.zeros(rows, 6),
            "valid_fraction": torch.ones(rows, 6),
            "role_index": torch.arange(6),
            "role_names": (
                "t0",
                "prev1",
                "prev2",
                "prev3",
                "seasonal",
                "year",
            ),
            "t0_index": 0,
            "ids": ["a", "b", "c"],
            "plume_ids": ["pa", "pb", "pc"],
            "event_ids": ["ea", "eb", "ec"],
        }
        base = {**common, "features": base_features}
        p5 = {
            **common,
            "features": torch.cat(
                (base_features, torch.randn(rows, 6, 768)), dim=-1
            ),
        }
        audit = target.audit_cache_alignment(base, p5)
        self.assertTrue(audit["p5_first_768_exact"])
        p5["features"][1, 2, 4] += 1
        with self.assertRaisesRegex(ValueError, "first 768"):
            target.audit_cache_alignment(base, p5)

    def test_patch_packet_is_identity_bound_and_fixed_weighted(self) -> None:
        rows = {
            "ids": ["a", "b"],
            "event_ids": ["ea", "eb"],
            "labels": torch.tensor([0, 1]),
        }
        component = {
            "logit_columns": ["seed17", "seed42"],
            "within_family_weights": [0.25, 0.75],
        }
        frame = pd.DataFrame(
            {
                "id": ["a", "b"],
                "event_id": ["ea", "eb"],
                "label": [0, 1],
                "seed17": [0.0, 2.0],
                "seed42": [2.0, 0.0],
            }
        )
        actual = target.patch_logits_from_packet(
            frame, rows=rows, component=component
        )
        np.testing.assert_array_equal(actual, np.asarray([1.5, 0.5]))
        frame.loc[1, "event_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "event IDs"):
            target.patch_logits_from_packet(
                frame, rows=rows, component=component
            )

    def test_bad_token_fails_before_any_outer_path_access(self) -> None:
        args = argparse.Namespace(
            confirm="wrong",
            lock_manifest="/path/that/does/not/exist",
            authorized_lock_sha256="none",
            base_cache="/outer/does/not/exist",
            p5_cache="/outer/does/not/exist",
            patch_packet=None,
            output_dir="/outer/does/not/exist",
            staging_dir="/outer/does/not/exist",
            batch_size=1,
            torch_threads=1,
        )
        with self.assertRaisesRegex(RuntimeError, "authorization token"):
            target.command_evaluate_once(args)

    def test_template_cannot_be_used_for_outer_evaluation(self) -> None:
        protocol = {"placeholder": True}
        manifest = {
            "status": "development_template_not_authorizable",
            "protocol": protocol,
            "protocol_sha256": target.canonical_digest(protocol),
            "test_or_sealed_or_holdout_read": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "LOCK_MANIFEST.json"
            target.atomic_json(path, manifest)
            with self.assertRaisesRegex(
                RuntimeError, "template cannot read outer"
            ):
                target.validate_lock(path, require_final=True)


if __name__ == "__main__":
    unittest.main()
