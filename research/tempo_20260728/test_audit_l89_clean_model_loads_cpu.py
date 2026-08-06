#!/usr/bin/env python3
"""CPU path-boundary tests for the clean model-load ledger."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.tempo_20260728 import audit_l89_clean_model_loads as target
from research.pretraining_20260727 import l89_ragged_cls_experiment as base


class ModelLoadLedgerTests(unittest.TestCase):
    def test_formal_artifact_must_stay_within_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="model_ledger_unit_") as name:
            root = Path(name) / "formal"
            root.mkdir()
            target.require_within(root / "checkpoint.pt", root, purpose="unit")
            with self.assertRaisesRegex(ValueError, "escapes"):
                target.require_within(
                    Path(name) / "old_checkpoint.pt", root, purpose="unit"
                )

    def test_held_out_marker_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "held-out"):
            target.safe_path("/tmp/outer_eval/model.pt", purpose="unit")

    def test_planned_artifacts_all_live_under_formal_root(self) -> None:
        root = Path("/tmp/clean_formal_unit")
        for paths in target.planned_paths(root).values():
            for path in paths:
                target.require_within(Path(path), root, purpose="unit")

    def test_exact_audited_cache_rejects_alias_and_stale_sha(self) -> None:
        with tempfile.TemporaryDirectory(prefix="model_ledger_unit_") as name:
            root = Path(name).resolve()
            expected = root / "fresh.pt"
            alias = root / "copied.pt"
            expected.write_bytes(b"fresh")
            alias.write_bytes(b"fresh")
            good = {
                "train_cache": str(expected),
                "train_cache_sha256": base.sha256_file(expected),
            }
            receipt = target.require_exact_audited_cache(
                good,
                key="train_cache",
                expected=expected,
                formal_root=root,
                purpose="unit cache",
            )
            self.assertEqual(receipt["sha256"], base.sha256_file(expected))

            wrong_path = dict(good, train_cache=str(alias))
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                target.require_exact_audited_cache(
                    wrong_path,
                    key="train_cache",
                    expected=expected,
                    formal_root=root,
                    purpose="unit cache",
                )

            stale = dict(good, train_cache_sha256="stale")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                target.require_exact_audited_cache(
                    stale,
                    key="train_cache",
                    expected=expected,
                    formal_root=root,
                    purpose="unit cache",
                )


if __name__ == "__main__":
    unittest.main()
