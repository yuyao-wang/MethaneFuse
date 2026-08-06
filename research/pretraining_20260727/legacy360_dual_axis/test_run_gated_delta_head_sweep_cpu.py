#!/usr/bin/env python3
"""CPU/bash contract tests for the gated-delta dev-only sweep launcher."""

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
LAUNCHER = SCRIPT_DIR / "run_gated_delta_head_sweep.sh"


def _cache(path: Path, *, split: str, encoder: str = "same") -> None:
    rows = 6
    encoder_payload = {"state_sha256": encoder}
    payload = {
        "schema_version": "query360-two-axis-feature-cache-v1",
        "split": split,
        "features": torch.zeros(rows, 4, 3, 8, dtype=torch.float16),
        "valid_mask": torch.ones(rows, 4, 3, dtype=torch.bool),
        "labels": torch.tensor([0, 1, 0, 1, 0, 1]),
        "encoder": encoder_payload,
        "sealed_test_read": False,
    }
    torch.save(payload, path)
    cache_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    encoder_fingerprint = hashlib.sha256(
        json.dumps(
            encoder_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
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
                "encoder_fingerprint": encoder_fingerprint,
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
                position = argv.index(option)
                return argv[position + 1]

            output = Path(value("--output-dir"))
            output.mkdir(parents=True, exist_ok=True)
            base_mode = value("--base-mode")
            arm = value("--arm")
            epochs = int(value("--epochs"))
            learning_rate = float(value("--learning-rate"))
            sensor_aux = float(value("--sensor-aux-weight"))
            bottleneck = int(value("--bottleneck-dim"))
            residual_cap = float(value("--residual-cap"))

            def write_json(name, payload):
                (output / name).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )

            write_json(
                "invocation.json",
                {"argv": argv},
            )
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
                    "model": {
                        "base_mode": base_mode,
                        "arm": arm,
                        "config": {
                            "bottleneck_dim": bottleneck,
                            "residual_cap": residual_cap,
                        },
                    },
                    "training": {
                        "epochs_requested": epochs,
                        "learning_rate": learning_rate,
                        "sensor_aux_weight": sensor_aux,
                    },
                },
            )
            write_json(
                "selection_lock.json",
                {
                    "test_cache_read_before_lock": False,
                    "dev_only_runner": True,
                    "base_mode": base_mode,
                    "arm": arm,
                },
            )
            (output / "checkpoint_best.pth").write_bytes(b"synthetic")
            """
        ),
        encoding="utf-8",
    )


class GatedDeltaSweepLauncherTests(unittest.TestCase):
    def _base_env(self, root: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "FORMAL_ROOT": str(root / "formal"),
                "HEADS_ROOT": str(root / "heads"),
                "GPU_IDS": "0,1",
                "MAX_HEADS_PER_GPU": "1",
                "EPOCHS": "3",
                "BATCH_SIZE": "4",
                "EVAL_BATCH_SIZE": "8",
            }
        )
        return environment

    def _run(
        self,
        arguments: list[str],
        *,
        environment: dict[str, str],
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

    def test_bash_syntax_and_default_dry_run_has_exact_grid(self) -> None:
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
            environment = self._base_env(root)
            result = self._run([], environment=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry-run only", result.stdout)
            self.assertFalse((root / "heads").exists())
            rows = [
                line.split()
                for line in result.stdout.splitlines()
                if line.startswith("gdelta_")
            ]
            self.assertEqual(len(rows), 8)
            observed = {
                (row[1], float(row[2]), float(row[3])) for row in rows
            }
            expected = {
                (base, learning_rate, residual_cap)
                for base, learning_rate, residual_cap in itertools.product(
                    ("hybrid", "universal"),
                    (1e-4, 3e-4),
                    (1.5, 4.0),
                )
            }
            self.assertEqual(observed, expected)

    def test_epoch_gpu_and_heldout_path_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for epochs in ("1", "2", "3"):
                environment = self._base_env(root)
                environment["EPOCHS"] = epochs
                result = self._run(["--plan"], environment=environment)
                self.assertEqual(result.returncode, 0, result.stderr)
            for epochs in ("0", "4", "10", "-1"):
                environment = self._base_env(root)
                environment["EPOCHS"] = epochs
                result = self._run(["--plan"], environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("EPOCHS must be 1, 2, or 3", result.stderr)

            environment = self._base_env(root)
            environment["GPU_IDS"] = "0,0"
            result = self._run(["--plan"], environment=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not contain duplicates", result.stderr)

            environment = self._base_env(root)
            environment["DEV_CACHE"] = str(root / "sealed_dev.pt")
            result = self._run(["--plan"], environment=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("looks held out", result.stderr)
            self.assertFalse((root / "heads").exists())

    def test_synthetic_run_writes_eight_isolated_dev_only_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train_core.pt"
            dev = root / "dev.pt"
            fake_runner = root / "fake_runner.py"
            _cache(train, split="train_core")
            _cache(dev, split="dev")
            _fake_runner(fake_runner)

            environment = self._base_env(root)
            environment.update(
                {
                    "TRAIN_CACHE": str(train),
                    "DEV_CACHE": str(dev),
                    "RUNNER": str(fake_runner),
                    "EPOCHS": "2",
                }
            )
            result = self._run(["--run"], environment=environment)
            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn(
                "All 8 development-only gated-delta pilots completed",
                result.stdout,
            )

            heads = root / "heads"
            run_dirs = sorted(
                path for path in heads.iterdir() if path.is_dir()
            )
            self.assertEqual(len(run_dirs), 8)
            observed: set[tuple[str, float, float]] = set()
            gpu_counts: dict[str, int] = {"cuda:0": 0, "cuda:1": 0}
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
                self.assertEqual(
                    option("--arm"), "scale_aware_gated_delta"
                )
                self.assertEqual(option("--epochs"), "2")
                self.assertEqual(option("--bottleneck-dim"), "32")
                self.assertEqual(option("--sensor-aux-weight"), "0.1")
                observed.add(
                    (
                        option("--base-mode"),
                        float(option("--learning-rate")),
                        float(option("--residual-cap")),
                    )
                )
                gpu_counts[option("--device")] += 1

                launcher_status = json.loads(
                    (run_dir / "launcher_status.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(launcher_status["status"], "complete")
                self.assertTrue(launcher_status["dev_only"])
                self.assertFalse(launcher_status["sealed_test_read"])
                self.assertEqual(
                    launcher_status["sealed_test_evaluations"], 0
                )
                log = (run_dir / "stdout_stderr.log").read_text(
                    encoding="utf-8"
                )
                self.assertIn("final_state=complete", log)
                self.assertFalse(
                    (run_dir / "sealed_test_result.json").exists()
                )

            expected = set(
                itertools.product(
                    ("hybrid", "universal"),
                    (1e-4, 3e-4),
                    (1.5, 4.0),
                )
            )
            self.assertEqual(observed, expected)
            self.assertEqual(gpu_counts, {"cuda:0": 4, "cuda:1": 4})

            repeated = self._run(["--run"], environment=environment)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn(
                "Refusing to reuse existing run directory",
                repeated.stderr,
            )

    def test_preflight_rejects_split_or_encoder_mismatch_before_outputs(
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
            environment = self._base_env(root)
            environment.update(
                {
                    "TRAIN_CACHE": str(train),
                    "DEV_CACHE": str(dev),
                    "RUNNER": str(fake_runner),
                }
            )
            mismatch = self._run(["--run"], environment=environment)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn(
                "train/dev encoder provenance differs", mismatch.stderr
            )
            self.assertFalse((root / "heads").exists())

            _cache(dev, split="test", encoder="encoder-a")
            split_swap = self._run(["--run"], environment=environment)
            self.assertNotEqual(split_swap.returncode, 0)
            self.assertIn("dev cache split='test'", split_swap.stderr)
            self.assertFalse((root / "heads").exists())


if __name__ == "__main__":
    unittest.main()
