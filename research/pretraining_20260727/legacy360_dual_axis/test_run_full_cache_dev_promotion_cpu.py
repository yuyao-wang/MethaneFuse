#!/usr/bin/env python3
"""CPU/bash smoke tests for the four-arm full-cache promotion launcher."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "run_full_cache_dev_promotion.sh"


def _fingerprint(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _cache(path: Path, *, split: str, rows: int) -> None:
    path.write_bytes(f"synthetic-{split}-{rows}".encode("utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    encoder = {"state_sha256": "a" * 64}
    if split == "train_core":
        audit = {
            "schema_version": "legacy360-merged-feature-cache-audit-v1",
            "split": split,
            "output": {
                "path": str(path.absolute()),
                "sha256": digest,
                "rows": rows,
            },
            "compatibility": {
                "encoder_fingerprint": _fingerprint(encoder),
                "tensor_fields": {
                    "features": {
                        "shape": [rows, 4, 3, 768],
                        "dtype": "torch.float16",
                    }
                },
            },
            "inputs": [{"part": index} for index in range(15)],
            "integrity": {
                "ids_unique": True,
                "query360_indices_unique": True,
                "plumes_disjoint_between_inputs": True,
            },
        }
    else:
        audit = {
            "schema_version": "query360-two-axis-feature-cache-v1",
            "split": split,
            "cache": str(path.absolute()),
            "cache_sha256": digest,
            "rows": rows,
            "feature_shape": [rows, 4, 3, 768],
            "encoder": encoder,
            "extraction": {"sealed_test_authorized": False},
        }
    Path(str(path) + ".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fake_runner(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import hashlib
            import json
            import sys
            from pathlib import Path

            argv = sys.argv[1:]
            if not argv or argv[0] != "train":
                raise SystemExit("fake runner accepts only train")

            def value(option):
                return argv[argv.index(option) + 1]

            output = Path(value("--output-dir")).absolute()
            output.mkdir(parents=True, exist_ok=True)
            arm = value("--arm")
            base_mode = value("--base-mode")
            epochs = int(value("--epochs"))
            learning_rate = float(value("--learning-rate"))
            gated = "--bottleneck-dim" in argv
            if gated:
                model_config = {
                    "feature_dim": 8,
                    "num_sensors": 4,
                    "num_roles": 3,
                    "bottleneck_dim": int(value("--bottleneck-dim")),
                    "dropout": float(value("--dropout")),
                    "residual_cap": float(value("--residual-cap")),
                }
                training = {
                    "epochs_requested": epochs,
                    "learning_rate": learning_rate,
                }
            else:
                model_config = {
                    "embed_dim": 8,
                    "num_sensors": 4,
                    "num_roles": 3,
                    "model_dim": int(value("--model-dim")),
                    "num_heads": int(value("--num-heads")),
                    "temporal_depth": int(value("--temporal-depth")),
                    "dropout": float(value("--dropout")),
                }
                training = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                }

            checkpoint = output / "checkpoint_best.pth"
            checkpoint.write_bytes(
                f"synthetic:{arm}:{learning_rate}".encode("utf-8")
            )
            checkpoint_sha = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            history = [
                {
                    "epoch": 0,
                    "train": None,
                    "dev": {"best_binary_f1": 0.89},
                    "interpretation": "exact unchanged checkpoint base",
                }
            ]
            lock = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "best_epoch": 0,
                "base_mode": base_mode,
                "arm": arm,
                "model_config": model_config,
                "encoder": {"state_sha256": "a" * 64},
                "test_cache_read_before_lock": False,
            }

            def write_json(name, payload):
                (output / name).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )

            write_json("invocation.json", {"argv": argv})
            write_json("metrics_history.json", history)
            write_json(
                "run_status.json",
                {
                    "status": "complete",
                    "dev_only": True,
                    "sealed_test_read": False,
                    "sealed_test_evaluations": 0,
                },
            )
            write_json(
                "summary.json",
                {
                    "status": "complete",
                    "sealed_test_read": False,
                    "sealed_test_evaluations": 0,
                    "selection_lock": lock,
                    "history": history,
                    "model": {
                        "base_mode": base_mode,
                        "arm": arm,
                    },
                    "training": training,
                },
            )
            write_json("selection_lock.json", lock)
            """
        ),
        encoding="utf-8",
    )


class FullCachePromotionLauncherTests(unittest.TestCase):
    def _environment(self, root: Path) -> dict[str, str]:
        formal = root / "formal"
        features = formal / "features"
        features.mkdir(parents=True, exist_ok=True)
        return {
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "FORMAL_ROOT": str(formal),
            "TRAIN_CACHE": str(
                features / "train_core_universal_s2hybrid.pt"
            ),
            "DEV_CACHE": str(features / "dev_universal_s2hybrid.pt"),
            "HEADS_ROOT": str(root / "promotion_heads"),
            "GPU_IDS": "0,1",
            "MAX_HEADS_PER_GPU": "1",
            "EPOCHS": "2",
            "BATCH_SIZE": "4",
            "EVAL_BATCH_SIZE": "8",
        }

    def _run(
        self,
        args: list[str],
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            cwd=LAUNCHER.parents[3],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )

    def test_bash_syntax_and_default_exact_dry_run(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("--test-cache", source)
        self.assertNotIn("--sealed-test", source)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = self._environment(root)
            result = self._run([], environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry-run only", result.stdout)
            self.assertIn("epoch-0 exact PTH base remains eligible", result.stdout)
            self.assertFalse(Path(environment["HEADS_ROOT"]).exists())
            rows = [
                line.split()
                for line in result.stdout.splitlines()
                if line.startswith("promote_")
            ]
            self.assertEqual(len(rows), 4)
            observed = {
                (row[1], row[2], float(row[3]), row[4]) for row in rows
            }
            self.assertEqual(
                observed,
                {
                    (
                        "gated_delta",
                        "scale_aware_gated_delta",
                        1e-4,
                        "cap1.5_b32",
                    ),
                    (
                        "gated_delta",
                        "scale_aware_gated_delta",
                        3e-4,
                        "cap1.5_b32",
                    ),
                    (
                        "compact_axial",
                        "scale_aware_two_axis_query",
                        1e-4,
                        "d64_depth1_drop0",
                    ),
                    (
                        "compact_axial",
                        "two_axis_query",
                        3e-5,
                        "d64_depth1_drop0",
                    ),
                },
            )

    def test_epoch_gpu_and_full_cache_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for valid in ("1", "2", "3"):
                environment = self._environment(root)
                environment["EPOCHS"] = valid
                result = self._run(["--plan"], environment)
                self.assertEqual(result.returncode, 0, result.stderr)
            for invalid in ("0", "4", "10"):
                environment = self._environment(root)
                environment["EPOCHS"] = invalid
                result = self._run(["--plan"], environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("EPOCHS must be 1, 2, or 3", result.stderr)

            environment = self._environment(root)
            environment["GPU_IDS"] = "0,0"
            result = self._run(["--plan"], environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not contain duplicates", result.stderr)

            environment = self._environment(root)
            environment["TRAIN_CACHE"] = str(
                root
                / "pilot"
                / "train_core_universal_s2hybrid.pt"
            )
            result = self._run(["--plan"], environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing non-full", result.stderr)

    def test_fake_run_dispatches_four_isolated_candidates_and_refuses_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = self._environment(root)
            train = Path(environment["TRAIN_CACHE"])
            dev = Path(environment["DEV_CACHE"])
            _cache(train, split="train_core", rows=113843)
            _cache(dev, split="dev", rows=12621)
            fake = root / "fake_runner.py"
            _fake_runner(fake)
            environment["GATED_RUNNER"] = str(fake)
            environment["TWO_AXIS_RUNNER"] = str(fake)

            result = self._run(["--run"], environment)
            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn(
                "All 4 complete-cache development-only promotion "
                "candidates completed",
                result.stdout,
            )
            heads = Path(environment["HEADS_ROOT"])
            campaign = json.loads(
                (heads / "promotion_launcher_status.json").read_text()
            )
            self.assertEqual(campaign["status"], "complete")
            self.assertEqual(campaign["candidate_count"], 4)
            self.assertTrue(campaign["epoch0_base_required"])
            run_dirs = sorted(path for path in heads.iterdir() if path.is_dir())
            self.assertEqual(len(run_dirs), 4)

            observed: set[tuple[str, str, float]] = set()
            gpu_counts = {"cuda:0": 0, "cuda:1": 0}
            for run_dir in run_dirs:
                invocation = json.loads(
                    (run_dir / "invocation.json").read_text()
                )
                argv = invocation["argv"]

                def option(name: str) -> str:
                    return argv[argv.index(name) + 1]

                self.assertEqual(argv[0], "train")
                self.assertEqual(option("--base-mode"), "universal")
                self.assertEqual(option("--epochs"), "2")
                self.assertNotIn("--test-cache", argv)
                self.assertNotIn("--sealed-test", argv)
                arm = option("--arm")
                learning_rate = float(option("--learning-rate"))
                family = (
                    "gated_delta"
                    if "--bottleneck-dim" in argv
                    else "compact_axial"
                )
                observed.add((family, arm, learning_rate))
                gpu_counts[option("--device")] += 1
                if family == "gated_delta":
                    self.assertEqual(option("--bottleneck-dim"), "32")
                    self.assertEqual(option("--residual-cap"), "1.5")
                else:
                    self.assertEqual(option("--model-dim"), "64")
                    self.assertEqual(option("--temporal-depth"), "1")
                    self.assertEqual(option("--dropout"), "0.0")

                status = json.loads(
                    (run_dir / "launcher_status.json").read_text()
                )
                self.assertEqual(status["status"], "complete")
                self.assertTrue(status["dev_only"])
                self.assertTrue(status["epoch0_base_verified"])
                self.assertFalse(status["sealed_test_read"])
                self.assertEqual(status["sealed_test_evaluations"], 0)
                log = (run_dir / "stdout_stderr.log").read_text()
                self.assertIn("epoch0_verified=true", log)

            self.assertEqual(
                observed,
                {
                    (
                        "gated_delta",
                        "scale_aware_gated_delta",
                        1e-4,
                    ),
                    (
                        "gated_delta",
                        "scale_aware_gated_delta",
                        3e-4,
                    ),
                    (
                        "compact_axial",
                        "scale_aware_two_axis_query",
                        1e-4,
                    ),
                    ("compact_axial", "two_axis_query", 3e-5),
                },
            )
            self.assertEqual(gpu_counts, {"cuda:0": 2, "cuda:1": 2})

            repeated = self._run(["--run"], environment)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn(
                "Refusing existing promotion output root",
                repeated.stderr,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
