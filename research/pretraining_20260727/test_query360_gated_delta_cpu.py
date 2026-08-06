#!/usr/bin/env python3
"""Focused CPU tests for the low-capacity legacy-360 gated-delta path."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from query360_gated_delta_model import GatedDelta360Head  # noqa: E402
from query360_gated_delta_runner import (  # noqa: E402
    build_parser,
    guard_development_path,
    main as runner_main,
    sha256_file,
)
from query360_model import transient_query_loss  # noqa: E402


class GatedDelta360HeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.features = torch.randn(6, 4, 3, 16)
        self.valid = torch.tensor(
            [
                [[1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1]],
                [[1, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0]],
                [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
            ],
            dtype=torch.bool,
        )
        self.model = GatedDelta360Head(
            16, bottleneck_dim=8, dropout=0.0
        )

    def test_output_loss_and_axis_masks(self) -> None:
        output = self.model(
            self.features, self.valid, arm="gated_delta"
        )
        self.assertEqual(tuple(output.fused_logits.shape), (6,))
        self.assertEqual(tuple(output.sensor_logits.shape), (6, 4))
        self.assertEqual(tuple(output.history_attention.shape), (6, 4, 2))
        self.assertTrue(torch.isfinite(output.fused_logits).all())
        self.assertTrue(
            torch.allclose(
                output.sensor_attention.sum(dim=1),
                torch.ones(6),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                output.sensor_attention.masked_select(~output.sensor_valid),
                torch.zeros_like(
                    output.sensor_attention.masked_select(
                        ~output.sensor_valid
                    )
                ),
            )
        )
        history_sum = output.history_attention.sum(dim=-1)
        self.assertTrue(
            torch.allclose(
                history_sum.masked_select(output.history_valid),
                torch.ones_like(
                    history_sum.masked_select(output.history_valid)
                ),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                history_sum.masked_select(~output.history_valid),
                torch.zeros_like(
                    history_sum.masked_select(~output.history_valid)
                ),
            )
        )
        loss = transient_query_loss(
            output,
            torch.tensor([0, 1, 0, 1, 1, 0]),
            auxiliary_weight=0.1,
        )
        self.assertTrue(torch.isfinite(loss.total))

    def test_initialization_exactly_preserves_supplied_pth_logits(self) -> None:
        base_fused = torch.linspace(-3.0, 2.0, 6)
        base_sensor = torch.randn(6, 4)
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.features,
                self.valid,
                base_fused_logits=base_fused,
                base_sensor_logits=base_sensor,
            )
        self.assertTrue(torch.equal(output.fused_logits, base_fused))
        self.assertTrue(
            torch.equal(
                output.sensor_logits.masked_select(output.sensor_valid),
                base_sensor.masked_select(output.sensor_valid),
            )
        )
        self.assertTrue(
            torch.equal(
                output.residual_fused_logits,
                torch.zeros_like(output.residual_fused_logits),
            )
        )

    def test_invalid_payloads_cannot_change_evidence_or_gates(self) -> None:
        self.model.eval()
        changed = self.features.clone()
        changed[~self.valid] = float("nan")
        with torch.no_grad():
            first = self.model(
                self.features, self.valid, arm="gated_delta"
            )
            second = self.model(changed, self.valid, arm="gated_delta")
        self.assertTrue(
            torch.equal(first.sensor_evidence, second.sensor_evidence)
        )
        self.assertTrue(
            torch.equal(first.sensor_attention, second.sensor_attention)
        )
        self.assertTrue(
            torch.equal(first.history_attention, second.history_attention)
        )

    def test_valid_history_changes_temporal_axis(self) -> None:
        self.model.eval()
        changed = self.features.clone()
        changed[:, :, 1:, 0] += (
            self.valid[:, :, 1:].to(changed.dtype) * 10.0
        )
        with torch.no_grad():
            first = self.model(
                self.features, self.valid, arm="gated_delta"
            )
            second = self.model(changed, self.valid, arm="gated_delta")
        self.assertFalse(
            torch.equal(first.sensor_evidence, second.sensor_evidence)
        )
        no_history = ~first.history_valid
        self.assertTrue(
            torch.equal(
                first.temporal_gate.masked_select(no_history),
                torch.zeros_like(
                    first.temporal_gate.masked_select(no_history)
                ),
            )
        )

    def test_scale_aware_arm_masks_only_s5p_history(self) -> None:
        output = self.model(
            self.features,
            self.valid,
            arm="scale_aware_gated_delta",
        )
        self.assertFalse(output.effective_time_valid[:, 3, 1:].any())
        self.assertTrue(
            torch.equal(
                output.effective_time_valid[:, :3],
                self.valid[:, :3],
            )
        )

    def test_two_optimizer_steps_reach_temporal_transform(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.2)
        target = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float32)
        base = torch.linspace(-0.5, 0.5, 6)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(
                self.features,
                self.valid,
                arm="gated_delta",
                base_fused_logits=base,
            )
            loss = F.binary_cross_entropy_with_logits(
                output.fused_logits, target
            )
            loss.backward()
            optimizer.step()
        gradient = self.model.delta_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_default_model_is_low_capacity(self) -> None:
        model = GatedDelta360Head(768, bottleneck_dim=32)
        parameters = sum(value.numel() for value in model.parameters())
        self.assertEqual(parameters, 51288)
        self.assertLess(parameters, 60_000)

    def test_requires_at_least_one_current_sensor(self) -> None:
        bad = self.valid.clone()
        bad[0, :, 0] = False
        with self.assertRaisesRegex(ValueError, "current sensor"):
            self.model(self.features, bad)


class DevelopmentOnlyRunnerTests(unittest.TestCase):
    @staticmethod
    def _payload(
        *,
        split: str,
        rows: int,
        offset: int,
    ) -> dict:
        generator = torch.Generator().manual_seed(100 + offset)
        labels = torch.tensor(
            [(index + offset) % 2 for index in range(rows)],
            dtype=torch.long,
        )
        features = torch.randn(rows, 4, 3, 12, generator=generator)
        valid = torch.ones(rows, 4, 3, dtype=torch.bool)
        valid[::3, 2] = False
        valid[1::4, 3, 1:] = False
        sign = labels.float() * 2.0 - 1.0
        base = sign * 1.5
        sensor_base = sign[:, None].repeat(1, 4)
        return {
            "schema_version": "query360-two-axis-feature-cache-v1",
            "split": split,
            "features": features.half(),
            "features_hybrid": features.half(),
            "features_universal": features.half(),
            "valid_mask": valid,
            "labels": labels,
            "base_fused_logits": base.clone(),
            "base_sensor_logits": sensor_base.clone(),
            "base_hybrid_logits": base.clone(),
            "base_universal_logits": base.clone(),
            "base_sensor_logits_hybrid": sensor_base.clone(),
            "base_sensor_logits_universal": sensor_base.clone(),
            "ids": [f"{split}-row-{offset + i}" for i in range(rows)],
            "plume_ids": [
                f"{split}-plume-{offset + i}" for i in range(rows)
            ],
            "event_ids": [
                f"{split}-event-{offset + i}" for i in range(rows)
            ],
            "availability_signatures": ["s2+l89+emit+s5p"] * rows,
            "encoder": {"state_sha256": "same-encoder"},
            "manifest": {
                "path": f"/synthetic/{split}.csv",
                "sha256": f"{split}-sha",
                "rows": rows,
            },
        }

    def _fit_once(self, root: Path) -> tuple[Path, Path, Path]:
        train_path = root / "train_core.pt"
        dev_path = root / "dev.pt"
        output_dir = root / "head"
        torch.save(
            self._payload(split="train_core", rows=24, offset=0),
            train_path,
        )
        torch.save(
            self._payload(split="dev", rows=12, offset=1000),
            dev_path,
        )
        runner_main(
            [
                "train",
                "--train-cache",
                str(train_path),
                "--dev-cache",
                str(dev_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--early-stop-patience",
                "0",
                "--batch-size",
                "12",
                "--eval-batch-size",
                "12",
                "--bottleneck-dim",
                "8",
                "--dropout",
                "0",
            ]
        )
        return (
            output_dir,
            output_dir / "checkpoint_best.pth",
            output_dir / "selection_lock.json",
        )

    def test_parser_exposes_no_test_or_sealed_input(self) -> None:
        parser = build_parser()
        destinations = {
            action.dest
            for action in parser._subparsers._group_actions[0]
            .choices["train"]
            ._actions
        }
        self.assertNotIn("test_cache", destinations)
        self.assertNotIn("sealed_test", destinations)
        locked_destinations = {
            action.dest
            for action in parser._subparsers._group_actions[0]
            .choices["evaluate-locked"]
            ._actions
        }
        self.assertIn("test_cache", locked_destinations)
        self.assertIn("sealed_test", locked_destinations)

    def test_guard_refuses_test_or_sealed_paths(self) -> None:
        with self.assertRaises(PermissionError):
            guard_development_path(
                Path("/tmp/formal/test_cache.pt"), role="train-cache"
            )
        with self.assertRaises(PermissionError):
            guard_development_path(
                Path("/tmp/sealed-cache.pt"), role="dev-cache"
            )
        guard_development_path(
            Path("/tmp/formal/dev_cache.pt"), role="dev-cache"
        )

    def test_one_epoch_cpu_smoke_writes_dev_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir, checkpoint_path, _ = self._fit_once(root)
            with (output_dir / "selection_lock.json").open(
                "r", encoding="utf-8"
            ) as stream:
                lock = json.load(stream)
            with (output_dir / "summary.json").open(
                "r", encoding="utf-8"
            ) as stream:
                summary = json.load(stream)
            self.assertTrue(lock["dev_only_runner"])
            self.assertFalse(lock["test_cache_read_before_lock"])
            self.assertFalse(summary["sealed_test_read"])
            self.assertEqual(summary["sealed_test_evaluations"], 0)
            self.assertIn(summary["best"]["epoch"], (0, 1))
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["encoder"], lock["encoder"])

    def test_locked_test_requires_authorization_runs_once_and_never_searches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, checkpoint_path, lock_path = self._fit_once(root)
            test_path = root / "sealed_test.pt"
            torch.save(
                self._payload(split="test", rows=14, offset=2000),
                test_path,
            )

            denied_output = root / "denied"
            with self.assertRaises(PermissionError):
                runner_main(
                    [
                        "evaluate-locked",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--selection-lock",
                        str(lock_path),
                        "--test-cache",
                        str(test_path),
                        "--output-dir",
                        str(denied_output),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertFalse(denied_output.exists())

            output_dir = root / "locked_eval"
            evaluate_args = [
                "evaluate-locked",
                "--checkpoint",
                str(checkpoint_path),
                "--selection-lock",
                str(lock_path),
                "--test-cache",
                str(test_path),
                "--output-dir",
                str(output_dir),
                "--sealed-test",
                "--eval-batch-size",
                "7",
                "--device",
                "cpu",
            ]
            # Development metrics are the only route that searches an F1
            # threshold. A locked evaluation must not invoke that route.
            with patch(
                "query360_gated_delta_runner.stratified_metrics",
                side_effect=AssertionError("test threshold search attempted"),
            ):
                runner_main(evaluate_args)

            with lock_path.open("r", encoding="utf-8") as stream:
                lock = json.load(stream)
            with (output_dir / "sealed_test_result.json").open(
                "r", encoding="utf-8"
            ) as stream:
                result = json.load(stream)
            with (output_dir / "locked_eval_status.json").open(
                "r", encoding="utf-8"
            ) as stream:
                status = json.load(stream)

            self.assertEqual(result["evaluation_count"], 1)
            self.assertFalse(result["test_threshold_search_performed"])
            self.assertFalse(
                result["metrics"]["test_threshold_search_performed"]
            )
            self.assertEqual(
                result["metrics"]["decision_threshold"],
                lock["locked_threshold"],
            )
            self.assertEqual(
                result["metrics"]["threshold_source"],
                "locked development threshold",
            )

            def nested_keys(value: object) -> set[str]:
                if isinstance(value, dict):
                    return set(value) | {
                        key
                        for child in value.values()
                        for key in nested_keys(child)
                    }
                if isinstance(value, list):
                    return {
                        key
                        for child in value
                        for key in nested_keys(child)
                    }
                return set()

            self.assertFalse(
                {
                    key
                    for key in nested_keys(result["metrics"])
                    if key.startswith("best_")
                }
            )
            self.assertTrue(status["sealed_test_read"])
            self.assertEqual(status["sealed_test_evaluations"], 1)
            with self.assertRaises(FileExistsError):
                runner_main(evaluate_args)

    def test_locked_contract_rejects_sha_config_base_arm_and_encoder_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, checkpoint_path, lock_path = self._fit_once(root)
            original_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            original_lock = json.loads(lock_path.read_text(encoding="utf-8"))

            cases = {
                "sha": ("checkpoint SHA", None),
                "model_config": ("model_config", "model_config"),
                "base_mode": ("base_mode", "base_mode"),
                "arm": ("arm", "arm"),
                "encoder": ("encoder provenance", "encoder"),
            }
            for name, (message, field) in cases.items():
                case_checkpoint = root / f"{name}.pth"
                case_lock = root / f"{name}.json"
                payload = dict(original_checkpoint)
                lock = dict(original_lock)
                if field == "model_config":
                    payload[field] = {
                        **payload[field],
                        "bottleneck_dim": int(
                            payload[field]["bottleneck_dim"]
                        )
                        + 1,
                    }
                elif field == "base_mode":
                    payload[field] = (
                        "universal"
                        if payload[field] == "hybrid"
                        else "hybrid"
                    )
                elif field == "arm":
                    payload[field] = (
                        "gated_delta"
                        if payload[field] == "scale_aware_gated_delta"
                        else "scale_aware_gated_delta"
                    )
                elif field == "encoder":
                    payload[field] = {"state_sha256": "tampered"}
                else:
                    payload["epoch"] = int(payload["epoch"]) + 1
                torch.save(payload, case_checkpoint)
                if name != "sha":
                    lock["checkpoint_sha256"] = sha256_file(case_checkpoint)
                case_lock.write_text(
                    json.dumps(lock), encoding="utf-8"
                )
                output_dir = root / f"{name}_eval"
                with self.subTest(case=name), self.assertRaisesRegex(
                    ValueError, message
                ):
                    runner_main(
                        [
                            "evaluate-locked",
                            "--checkpoint",
                            str(case_checkpoint),
                            "--selection-lock",
                            str(case_lock),
                            "--test-cache",
                            str(root / "must_not_be_read.pt"),
                            "--output-dir",
                            str(output_dir),
                            "--sealed-test",
                            "--device",
                            "cpu",
                        ]
                    )
                status = json.loads(
                    (output_dir / "locked_eval_status.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse(status["sealed_test_read"])

    def test_locked_test_encoder_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, checkpoint_path, lock_path = self._fit_once(root)
            payload = self._payload(split="test", rows=12, offset=3000)
            payload["encoder"] = {"state_sha256": "different-encoder"}
            test_path = root / "sealed_test.pt"
            torch.save(payload, test_path)
            with self.assertRaisesRegex(ValueError, "sealed-test encoder"):
                runner_main(
                    [
                        "evaluate-locked",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--selection-lock",
                        str(lock_path),
                        "--test-cache",
                        str(test_path),
                        "--output-dir",
                        str(root / "encoder_mismatch"),
                        "--sealed-test",
                        "--device",
                        "cpu",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
