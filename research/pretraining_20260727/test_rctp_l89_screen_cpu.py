#!/usr/bin/env python3
"""CPU regression tests for the narrow L89 RCTP mechanism screen."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import rctp_l89_screen as runner


class RCTPL89ScreenTests(unittest.TestCase):
    @staticmethod
    def _payload(rows: int = 24, timepoints: int = 6, feature_dim: int = 16):
        labels = torch.zeros(rows, dtype=torch.long)
        labels[-4:] = 1
        unique = torch.ones(rows, timepoints, dtype=torch.bool)
        unique[0, 2] = False
        return {
            "labels": labels,
            "unique_mask": unique,
            "features": torch.randn(rows, timepoints, feature_dim),
            "delta_days": torch.tensor(
                [[0.0, -8.0, -16.0, -24.0, -90.0, -365.0]]
            ).repeat(rows, 1),
            "role_index": torch.arange(timepoints),
        }

    def test_01_plan_is_deterministic_balanced_and_negative_only(self) -> None:
        payload = self._payload()
        config = runner.RendererConfig()
        first = runner.deterministic_reference_plan(
            payload,
            seed=9,
            epoch=0,
            max_rows=18,
            reference_label="negative",
            renderer=config,
        )
        second = runner.deterministic_reference_plan(
            payload,
            seed=9,
            epoch=0,
            max_rows=18,
            reference_label="negative",
            renderer=config,
        )
        self.assertEqual(first, second)
        self.assertEqual(runner.plan_sha256(first), runner.plan_sha256(second))
        self.assertEqual(len(first), 18)
        self.assertTrue(all(entry.row_index < 20 for entry in first))
        self.assertTrue(any(entry.visit_index == 0 for entry in first))
        self.assertTrue(any(entry.visit_index > 0 for entry in first))
        self.assertTrue(
            all(
                bool(payload["unique_mask"][entry.row_index, entry.visit_index])
                for entry in first
            )
        )
        self.assertTrue(
            all(
                config.min_peak_drop <= entry.peak_drop <= config.max_peak_drop
                for entry in first
            )
        )

    def test_02_soft_plume_is_deterministic_bounded_and_nontrivial(self) -> None:
        config = runner.RendererConfig()
        first = runner.make_soft_plume(32, 40, seed=123, config=config)
        second = runner.make_soft_plume(32, 40, seed=123, config=config)
        different = runner.make_soft_plume(32, 40, seed=124, config=config)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(tuple(first.shape), (32, 40))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertEqual(float(first.max()), 1.0)
        self.assertGreater(int((first > 0.1).sum()), 0)
        self.assertLess(int((first > 0.1).sum()), first.numel())

    def test_03_counterfactuals_are_exact_and_energy_matched(self) -> None:
        torch.manual_seed(4)
        batch, channels, height, width = 3, 7, 16, 16
        mean = torch.tensor([10.0, 11.0, 12.0, 13.0, 18.0, 19.0, 17.0])
        std = torch.tensor([2.0, 2.5, 3.0, 3.5, 3.0, 4.0, 4.5])
        raw = mean.view(1, -1, 1, 1) + torch.rand(
            batch, channels, height, width
        ) * std.view(1, -1, 1, 1)
        clean = (raw - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        valid = torch.ones_like(clean, dtype=torch.bool)
        valid[:, :, :2, :3] = False
        clean = torch.where(valid, clean, torch.zeros_like(clean))
        plume = torch.stack(
            [
                runner.make_soft_plume(
                    height, width, seed=10 + index, config=runner.RendererConfig()
                )
                for index in range(batch)
            ]
        )
        variants, diagnostics = runner.render_counterfactual_variants(
            clean,
            valid,
            plume,
            torch.tensor([0.01, 0.03, 0.08]),
            normalization_mean=mean,
            normalization_std=std,
            config=runner.RendererConfig(),
        )
        self.assertEqual(tuple(variants.shape), (batch, 3, channels, height, width))
        self.assertLess(float(diagnostics["max_energy_relative_error"]), 1e-5)
        energies = diagnostics["energies"]
        self.assertTrue(torch.allclose(energies, energies[:, :1], rtol=1e-5, atol=1e-7))
        deltas = diagnostics["deltas"]
        # Exact invalid-pixel control: no variant changes an invalid pixel.
        invalid = ~valid[:, None].expand_as(deltas)
        self.assertEqual(int(torch.count_nonzero(deltas[invalid])), 0)
        # Methane approximation leaves visible/NIR bands untouched and changes SWIR.
        self.assertEqual(int(torch.count_nonzero(deltas[:, 0, :5])), 0)
        self.assertGreater(float(deltas[:, 0, 6].abs().sum()), 0.0)
        self.assertGreater(
            float(deltas[:, 0, 6].abs().sum()),
            float(deltas[:, 0, 5].abs().sum()),
        )
        # Every arm is paired to the exact same clean tensor.
        self.assertTrue(
            torch.allclose(variants - clean[:, None], deltas, rtol=1e-5, atol=1e-7)
        )

    def test_04_batch_assembly_places_only_exact_delta_at_visit(self) -> None:
        torch.manual_seed(2)
        batch, timepoints, feature_dim = 4, 6, 8
        cached = torch.randn(batch, timepoints, feature_dim)
        valid = torch.ones(batch, timepoints, dtype=torch.bool)
        delta_days = torch.tensor(
            [[0.0, -1.0, -2.0, -3.0, -90.0, -365.0]]
        ).repeat(batch, 1)
        visits = torch.tensor([0, 1, 4, 5])
        clean = torch.randn(batch, feature_dim)
        variants = torch.randn(batch, 3, feature_dim)
        assembled = runner.assemble_probe_batch(
            cached, valid, delta_days, visits, clean, variants
        )
        self.assertEqual(
            tuple(assembled["delta_features"].shape),
            (batch * 3, timepoints, feature_dim),
        )
        self.assertEqual(
            assembled["type_target"].tolist(),
            [1.0, 0.0, 0.0] * batch,
        )
        for row in range(batch):
            for arm in range(3):
                flat = row * 3 + arm
                visit = int(visits[row])
                expected = variants[row, arm] - clean[row]
                self.assertTrue(
                    torch.equal(assembled["delta_features"][flat, visit], expected)
                )
                other = assembled["delta_features"][flat].clone()
                other[visit] = 0
                self.assertEqual(int(torch.count_nonzero(other)), 0)
                self.assertTrue(
                    torch.equal(assembled["clean_features"][flat, visit], clean[row])
                )

    def test_05_zero_init_contract_and_two_step_gradient(self) -> None:
        torch.manual_seed(12)
        batch, timepoints, feature_dim = 6, 6, 16
        response = runner.response_metadata(
            runner.DEFAULT_L89_CH4_RESPONSE,
            [442, 482, 560, 654, 864, 1608, 2203],
            timepoints=timepoints,
        )
        model = runner.RCTPTemporalProbe(
            feature_dim=feature_dim,
            response_dim=len(response),
            num_roles=timepoints,
            model_dim=24,
            num_heads=4,
            dropout=0.0,
        )
        clean = torch.randn(batch, timepoints, feature_dim)
        delta = torch.zeros_like(clean)
        delta[:, 2] = torch.randn(batch, feature_dim)
        valid = torch.ones(batch, timepoints, dtype=torch.bool)
        delta_days = torch.tensor(
            [[0.0, -1.0, -2.0, -3.0, -90.0, -365.0]]
        ).repeat(batch, 1)
        output = model(
            clean,
            delta,
            valid,
            torch.arange(timepoints),
            delta_days,
            response,
        )
        self.assertEqual(int(torch.count_nonzero(output["type_logit"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["visit_logits"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["strength_log"])), 0)
        self.assertEqual(int(torch.count_nonzero(model.adapter_out.weight)), 0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        for _ in range(2):
            output = model(
                clean,
                delta,
                valid,
                torch.arange(timepoints),
                delta_days,
                response,
            )
            loss = F.binary_cross_entropy_with_logits(
                output["type_logit"], targets
            ) + F.cross_entropy(
                output["visit_logits"], torch.full((batch,), 2)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.type_head.weight.abs().sum()), 0.0)
        self.assertGreater(float(model.visit_head.weight.abs().sum()), 0.0)
        self.assertGreater(float(model.adapter_out.weight.abs().sum()), 0.0)

    def test_06_metrics_preserve_matched_groups(self) -> None:
        labels = [1, 0, 0, 1, 0, 0]
        probabilities = [0.9, 0.2, 0.3, 0.8, 0.4, 0.1]
        metrics = runner.compute_screen_metrics(
            labels,
            probabilities,
            group_ids=[0, 0, 0, 1, 1, 1],
            variant_indices=[0, 1, 2, 0, 1, 2],
            visit_targets=[2] * 6,
            visit_predictions=[2] * 6,
            strength_targets=[-3.0] * 6,
            strength_predictions=[-3.0] * 6,
        )
        self.assertAlmostEqual(metrics["paired_win_rate"], 1.0)
        self.assertGreater(metrics["paired_probability_margin"], 0.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["visit_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["methane_strength_log_mae"], 0.0)

    def test_07_test_like_data_and_output_paths_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (
                root / "test.csv",
                root / "sealed" / "train.pt",
                root / "rctp-test-output",
            ):
                with self.assertRaises(ValueError):
                    runner.cache_runner.assert_not_sealed_path(
                        value, purpose="synthetic"
                    )

    def test_08_parser_exposes_bounded_screen_controls_only(self) -> None:
        parser = runner.build_parser()
        args = parser.parse_args(
            [
                "--output-dir",
                "/tmp/rctp-dev-output",
                "--max-train-rows",
                "128",
                "--max-dev-rows",
                "64",
                "--max-train-steps",
                "7",
            ]
        )
        self.assertEqual(args.max_train_rows, 128)
        self.assertEqual(args.max_dev_rows, 64)
        self.assertEqual(args.max_train_steps, 7)
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("test_csv", destinations)
        self.assertNotIn("test_cache", destinations)

    def test_09_tiny_epoch_path_emits_auditable_predictions(self) -> None:
        class TinyDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                config = runner.RendererConfig()
                return {
                    "row_index": torch.tensor(index, dtype=torch.long),
                    "visit_index": torch.tensor(index, dtype=torch.long),
                    "renderer_seed": torch.tensor(index + 20, dtype=torch.long),
                    "peak_drop": torch.tensor(0.02 + 0.01 * index),
                    "clean_image": torch.full((7, 8, 8), 0.2 + 0.1 * index),
                    "valid_pixels": torch.ones(7, 8, 8, dtype=torch.bool),
                    "plume_field": runner.make_soft_plume(
                        8, 8, seed=index + 20, config=config
                    ),
                }

        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "projection",
                    torch.arange(7 * 8, dtype=torch.float32).reshape(7, 8)
                    / 100.0,
                )

            def forward_features(self, x_dict):
                pooled = x_dict["imgs"].mean(dim=(2, 3))
                return {"x_norm_clstoken": pooled @ self.projection}

        payload = {
            "features": torch.randn(2, 6, 8),
            "unique_mask": torch.ones(2, 6, dtype=torch.bool),
            "delta_days": torch.tensor(
                [[0.0, -1.0, -2.0, -3.0, -90.0, -365.0]]
            ).repeat(2, 1),
            "role_index": torch.arange(6),
            "input_contract": {
                "normalization_mean": [10.0] * 7,
                "normalization_std": [2.0] * 7,
                "band_indices": list(range(7)),
                "channel_ids": [442, 482, 560, 654, 864, 1608, 2203],
            },
        }
        metadata = runner.response_metadata(
            runner.DEFAULT_L89_CH4_RESPONSE,
            payload["input_contract"]["channel_ids"],
            timepoints=6,
        )
        probe = runner.RCTPTemporalProbe(
            feature_dim=8,
            response_dim=len(metadata),
            num_roles=6,
            model_dim=12,
            num_heads=3,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        metrics, steps, predictions = runner._run_epoch(
            backbone=TinyBackbone(),
            probe=probe,
            loader=DataLoader(TinyDataset(), batch_size=2),
            payload=payload,
            renderer=runner.RendererConfig(),
            metadata=metadata,
            device=torch.device("cpu"),
            amp_dtype="float32",
            optimizer=optimizer,
            max_steps=0,
            global_step=0,
            type_weight=1.0,
            visit_weight=0.25,
            strength_weight=0.1,
            grad_clip=1.0,
            log_interval=20,
        )
        self.assertEqual(steps, 1)
        self.assertEqual(len(predictions), 6)
        self.assertEqual(
            predictions["variant"].tolist(),
            list(runner.VARIANT_NAMES) * 2,
        )
        self.assertEqual(
            predictions.groupby("cache_row_index").size().to_dict(), {0: 3, 1: 3}
        )
        self.assertTrue(np.isfinite(predictions["probability_methane"]).all())
        self.assertEqual(metrics["matched_groups"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
