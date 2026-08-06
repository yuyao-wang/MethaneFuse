from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_promoted_locked as wrapper  # noqa: E402
import sealed_test_feature_shards as sealed  # noqa: E402
import test_sealed_test_feature_shards_cpu as fixtures  # noqa: E402


class PromotedLockedEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="promoted-locked-eval-synthetic-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self, family: str) -> dict[str, Path]:
        family_fixture = (
            "gated" if family == "gated_delta" else "two_axis"
        )
        selection_dir = self.root / family / "selection"
        lock, checkpoint, lock_payload, _ = fixtures._write_selection(
            selection_dir,
            family=family_fixture,
        )
        case_root = self.root / family
        master = fixtures._write_master_selection(
            case_root,
            lock_path=lock,
            checkpoint_path=checkpoint,
            lock=lock_payload,
            family=family_fixture,
        )
        manifest = case_root / "sealed.csv"
        fixtures._write_manifest(
            manifest,
            [
                fixtures._row(1, event="e1", plume="p1"),
                fixtures._row(2, event="e2", plume="p2"),
                fixtures._row(3, event="e3", plume="p3"),
                fixtures._row(4, event="e4", plume="p4"),
            ],
        )
        shard_dir = case_root / "shards"
        sealed.command_shard_manifest(
            fixtures._shard_args(
                manifest=manifest,
                lock=lock,
                checkpoint=checkpoint,
                master=master,
                output=shard_dir,
            )
        )
        plan_path = shard_dir / "sealed_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        cache_paths: list[Path] = []
        for shard in plan["shards"]:
            cache_path = Path(shard["expected_cache_path"])
            fixtures._cache(
                cache_path,
                Path(shard["manifest_path"]),
                encoder=lock_payload["encoder"],
            )
            cache_paths.append(cache_path)
        merged = shard_dir / "merged.pt"
        merge_args = SimpleNamespace(
            input_cache=[str(path) for path in cache_paths],
            plan=str(plan_path),
            selection_lock=str(lock),
            checkpoint=str(checkpoint),
            master_selection_receipt=str(master),
            output_cache=str(merged),
            output_manifest="",
            audit="",
            receipt="",
            sealed_test=True,
        )
        sealed.command_merge_cache(merge_args)
        return {
            "master": master,
            "merge_receipt": merged.with_suffix(
                merged.suffix + ".receipt.json"
            ),
            "cache": merged,
            "lock": lock,
            "checkpoint": checkpoint,
            "output": case_root / "evaluation",
        }

    @staticmethod
    def _fake_evaluator(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess:
        del cwd, check

        def value(flag: str) -> str:
            return argv[argv.index(flag) + 1]

        output = Path(value("--output-dir"))
        output.mkdir(parents=True, exist_ok=True)
        lock_path = Path(value("--selection-lock"))
        checkpoint_path = Path(value("--checkpoint"))
        cache_path = Path(value("--test-cache"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        result_path = output / "sealed_test_result.json"
        sealed.atomic_json(
            result_path,
            {
                "artifact_type": "sealed_test_result",
                "evaluation_count": 1,
                "checkpoint": {
                    "sha256": sealed.sha256_file(checkpoint_path),
                },
                "selection_lock": {
                    "sha256": sealed.sha256_file(lock_path),
                    "locked_threshold": lock["locked_threshold"],
                },
                "cache": {"path": str(cache_path.absolute())},
                "metrics": {"binary_f1": 0.91},
                "test_threshold_search_performed": False,
                "test_cache_read_after_selection_lock": True,
            },
        )
        sealed.atomic_json(
            output / "locked_eval_status.json",
            {
                "status": "complete",
                "sealed_test_read": True,
                "sealed_test_evaluations": 1,
                "result": str(result_path),
            },
        )
        (output / "sealed_test_predictions.csv").write_text(
            "id,label,probability\nx,1,0.9\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0)

    def _args(
        self,
        prepared: dict[str, Path],
        *,
        output: Path | None = None,
        sealed_test: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            master_selection_receipt=str(prepared["master"]),
            merge_receipt=str(prepared["merge_receipt"]),
            test_cache=str(prepared["cache"]),
            output_dir=str(output or prepared["output"]),
            sealed_test=sealed_test,
            eval_batch_size=8,
            device="cpu",
            python=sys.executable,
        )

    def test_both_families_dispatch_once_from_master(self) -> None:
        for family, evaluator_name in (
            ("gated_delta", "query360_gated_delta_runner.py"),
            ("compact_axial", "query360_two_axis_full_legacy.py"),
        ):
            with self.subTest(family=family):
                prepared = self._prepare(family)
                with mock.patch.object(
                    wrapper,
                    "_preflight_family_model",
                ), mock.patch.object(
                    wrapper.subprocess,
                    "run",
                    side_effect=self._fake_evaluator,
                ) as runner:
                    wrapper.command_evaluate(self._args(prepared))
                argv = runner.call_args.args[0]
                self.assertTrue(argv[1].endswith(evaluator_name))
                receipt = json.loads(
                    (
                        prepared["output"] / "promotion_eval_receipt.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["master_winner_family"], family)
                self.assertEqual(receipt["sealed_test_evaluations"], 1)
                self.assertFalse(receipt["threshold_search_performed"])

                with mock.patch.object(
                    wrapper,
                    "_preflight_family_model",
                ), mock.patch.object(
                    wrapper,
                    "_validate_cache_after_claim",
                    side_effect=AssertionError("cache reopened"),
                ) as cache_reader:
                    with self.assertRaises(FileExistsError):
                        wrapper.command_evaluate(
                            self._args(
                                prepared,
                                output=self.root / family / "other-output",
                            )
                        )
                    cache_reader.assert_not_called()

    def test_explicit_flag_fails_before_any_receipt_or_cache_read(self) -> None:
        args = SimpleNamespace(
            master_selection_receipt="/must/not/open/master.json",
            merge_receipt="/must/not/open/merge.json",
            test_cache="/must/not/open/cache.pt",
            output_dir="/must/not/write",
            sealed_test=False,
            eval_batch_size=8,
            device="cpu",
            python=sys.executable,
        )
        with mock.patch.object(
            wrapper,
            "_validate_master_and_model",
            side_effect=AssertionError("opened"),
        ) as master_reader:
            with self.assertRaises(PermissionError):
                wrapper.command_evaluate(args)
            master_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
