#!/usr/bin/env python3
"""CPU/bash contracts for the ten-run two-axis dev-only launcher."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "run_two_axis_head_sweep.sh"


def _encoder_fingerprint(encoder: dict[str, Any]) -> str:
    canonical = json.dumps(
        encoder,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _cache(path: Path, *, split: str, encoder: str = "same") -> None:
    rows = 6
    encoder_payload = {"state_sha256": encoder}
    torch.save(
        {
            "schema_version": "query360-two-axis-feature-cache-v1",
            "split": split,
            "features": torch.zeros(rows, 4, 3, 8, dtype=torch.float16),
            "valid_mask": torch.ones(rows, 4, 3, dtype=torch.bool),
            "labels": torch.tensor([0, 1, 0, 1, 0, 1]),
            "encoder": encoder_payload,
            "sealed_test_read": False,
        },
        path,
    )
    cache_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if split == "train_core":
        audit: dict[str, Any] = {
            "schema_version": "legacy360-merged-feature-cache-audit-v1",
            "split": split,
            "output": {
                "path": str(path.absolute()),
                "sha256": cache_sha,
                "rows": rows,
            },
            "compatibility": {
                "encoder_fingerprint": _encoder_fingerprint(
                    encoder_payload
                ),
                "tensor_fields": {
                    "features": {
                        "shape": [rows, 4, 3, 8],
                        "dtype": "torch.float16",
                    }
                },
            },
        }
    else:
        audit = {
            "schema_version": "query360-two-axis-feature-cache-v1",
            "split": split,
            "cache": str(path.absolute()),
            "cache_sha256": cache_sha,
            "rows": rows,
            "feature_shape": [rows, 4, 3, 8],
            "encoder": encoder_payload,
            "extraction": {
                "sealed_test_authorized": split == "test",
            },
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
            import json
            import sys
            from pathlib import Path

            argv = sys.argv[1:]
            if not argv or argv[0] != "train":
                raise SystemExit("fake runner accepts only train")

            def value(option):
                return argv[argv.index(option) + 1]

            output = Path(value("--output-dir"))
            output.mkdir(parents=True, exist_ok=True)
            base_mode = value("--base-mode")
            arm = value("--arm")
            epochs = int(value("--epochs"))
            learning_rate = float(value("--learning-rate"))

            def write_json(name, payload):
                (output / name).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )

            write_json("invocation.json", {"argv": argv})
            write_json(
                "run_status.json",
                {
                    "status": "complete",
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
                    "model": {
                        "base_mode": base_mode,
                        "arm": arm,
                    },
                    "training": {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                    },
                },
            )
            write_json(
                "selection_lock.json",
                {
                    "test_cache_read_before_lock": False,
                    "base_mode": base_mode,
                    "arm": arm,
                },
            )
            (output / "checkpoint_best.pth").write_bytes(b"synthetic")
            """
        ),
        encoding="utf-8",
    )


class TwoAxisSweepLauncherTests(unittest.TestCase):
    def _environment(self, root: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "FORMAL_ROOT": str(root / "formal"),
                "HEADS_ROOT": str(root / "heads"),
                "GPU_IDS": "0,1",
                "MAX_HEADS_PER_GPU": "1",
                "BATCH_SIZE": "4",
                "EVAL_BATCH_SIZE": "8",
            }
        )
        return environment

    def _run(
        self, arguments: list[str], environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            cwd=LAUNCHER.parents[3],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )

    def test_current_torch_mmap_failure_is_removed_and_bash_is_valid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = Path(temporary) / "tiny.pt"
            torch.save({"value": torch.ones(1)}, payload)
            with self.assertRaisesRegex(TypeError, "mmap"):
                torch.load(
                    payload,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("mmap=True", source)
        self.assertIn("cache SHA/audit preflight passed", source)
        syntax = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_default_is_non_mutating_exact_ten_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._run([], self._environment(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry-run only", result.stdout)
            self.assertFalse((root / "heads").exists())
            rows = [
                line.split()
                for line in result.stdout.splitlines()
                if line.startswith(("hybrid_", "universal_"))
            ]
            self.assertEqual(len(rows), 10)
            observed = {
                (row[1], row[2], float(row[3])) for row in rows
            }
            expected = {
                ("hybrid", arm, learning_rate)
                for arm, learning_rate in itertools.product(
                    (
                        "current_only",
                        "two_axis_query",
                        "scale_aware_two_axis_query",
                    ),
                    (1e-4, 3e-4),
                )
            } | {
                ("universal", arm, learning_rate)
                for arm, learning_rate in itertools.product(
                    ("current_only", "scale_aware_two_axis_query"),
                    (1e-4, 3e-4),
                )
            }
            self.assertEqual(observed, expected)
            source = LAUNCHER.read_text(encoding="utf-8")
            self.assertNotIn("--test-cache", source)
            self.assertNotIn("--sealed-test", source)

    def test_synthetic_run_dispatches_ten_isolated_dev_only_jobs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train_core.pt"
            dev = root / "dev.pt"
            fake_runner = root / "fake_runner.py"
            _cache(train, split="train_core")
            _cache(dev, split="dev")
            _fake_runner(fake_runner)
            environment = self._environment(root)
            environment.update(
                {
                    "TRAIN_CACHE": str(train),
                    "DEV_CACHE": str(dev),
                    "RUNNER": str(fake_runner),
                }
            )
            result = self._run(["--run"], environment)
            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("All development-only head runs completed", result.stdout)
            run_dirs = sorted(
                path
                for path in (root / "heads").iterdir()
                if path.is_dir()
            )
            self.assertEqual(len(run_dirs), 10)
            gpu_counts = {"cuda:0": 0, "cuda:1": 0}
            combinations: set[tuple[str, str, float]] = set()
            for run_dir in run_dirs:
                invocation = json.loads(
                    (run_dir / "invocation.json").read_text(
                        encoding="utf-8"
                    )
                )
                argv = invocation["argv"]

                def option(name: str) -> str:
                    return argv[argv.index(name) + 1]

                self.assertEqual(argv[0], "train")
                self.assertNotIn("--test-cache", argv)
                self.assertNotIn("--sealed-test", argv)
                self.assertEqual(option("--epochs"), "3")
                combinations.add(
                    (
                        option("--base-mode"),
                        option("--arm"),
                        float(option("--learning-rate")),
                    )
                )
                gpu_counts[option("--device")] += 1
                status = json.loads(
                    (run_dir / "launcher_status.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(status["status"], "complete")
                self.assertTrue(status["dev_only"])
                self.assertFalse(status["sealed_test_read"])
                self.assertEqual(status["sealed_test_evaluations"], 0)
                self.assertIn(
                    "final_state=complete",
                    (run_dir / "stdout_stderr.log").read_text(
                        encoding="utf-8"
                    ),
                )
                self.assertFalse(
                    (run_dir / "sealed_test_result.json").exists()
                )
            self.assertEqual(len(combinations), 10)
            self.assertEqual(gpu_counts, {"cuda:0": 5, "cuda:1": 5})

            repeated = self._run(["--run"], environment)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn(
                "Refusing to reuse existing run directory",
                repeated.stderr,
            )

    def test_preflight_rejects_sha_split_and_encoder_before_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train_core.pt"
            dev = root / "dev.pt"
            fake_runner = root / "fake_runner.py"
            _cache(train, split="train_core", encoder="encoder-a")
            _cache(dev, split="dev", encoder="encoder-b")
            _fake_runner(fake_runner)
            environment = self._environment(root)
            environment.update(
                {
                    "TRAIN_CACHE": str(train),
                    "DEV_CACHE": str(dev),
                    "RUNNER": str(fake_runner),
                }
            )
            mismatch = self._run(["--run"], environment)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn(
                "train/dev encoder provenance differs", mismatch.stderr
            )
            self.assertFalse((root / "heads").exists())

            _cache(dev, split="dev", encoder="encoder-a")
            with dev.open("ab") as stream:
                stream.write(b"tampered")
            tampered = self._run(["--run"], environment)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("dev cache SHA mismatch", tampered.stderr)
            self.assertFalse((root / "heads").exists())

            _cache(dev, split="test", encoder="encoder-a")
            split_swap = self._run(["--run"], environment)
            self.assertNotEqual(split_swap.returncode, 0)
            self.assertIn("dev cache split='test'", split_swap.stderr)
            self.assertFalse((root / "heads").exists())


if __name__ == "__main__":
    unittest.main()
