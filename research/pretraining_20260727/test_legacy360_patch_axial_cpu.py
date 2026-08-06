#!/usr/bin/env python3
"""CPU contract tests for ``legacy360_patch_axial.py``."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from legacy360_patch_axial import (  # noqa: E402
    AuthorizedSealedQuery360Dataset,
    LegacyCLSHead,
    Legacy360PatchAxial,
    SEALED_TEST_AUTHORIZATION,
    _two_class_log_odds,
)
from query360_data import Query360Dataset, UnsafePathError  # noqa: E402


class TinyPatchBackbone(nn.Module):
    """Small deterministic encoder with the Panopticon output contract."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(4, embed_dim, bias=False)
        torch.manual_seed(7)
        nn.init.normal_(self.projection.weight, std=0.1)
        # Empty means the model uses the native forward_features path for both
        # the base and per-frame branches.
        self.blocks = nn.ModuleList()

    def forward_features(self, x_dict):
        images = x_dict["imgs"].float()
        pooled = torch.nn.functional.adaptive_avg_pool2d(images.mean(1, keepdim=True), (2, 2))
        patches = pooled.flatten(2).transpose(1, 2)
        # Each scalar patch is expanded into four simple moments.
        moments = torch.cat(
            (patches, patches.square(), patches.sin(), torch.ones_like(patches)),
            dim=-1,
        )
        patch_tokens = self.projection(moments)
        cls = patch_tokens.mean(dim=1)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patch_tokens,
        }


def sensor_batch(
    *,
    global_rows: list[int],
    roles: list[int],
    values: list[float],
    channels: int = 2,
) -> dict[str, torch.Tensor]:
    images = torch.stack(
        [
            torch.full((channels, 4, 4), float(value), dtype=torch.float32)
            for value in values
        ]
    )
    return {
        "images": images,
        "rows": torch.tensor(global_rows, dtype=torch.long),
        "roles": torch.tensor(roles, dtype=torch.long),
        "channel_ids": torch.arange(1, channels + 1, dtype=torch.int16),
    }


def empty_sensor_batch(channels: int = 2) -> dict[str, torch.Tensor]:
    return {
        "images": torch.empty(0, channels, 0, 0),
        "rows": torch.empty(0, dtype=torch.long),
        "roles": torch.empty(0, dtype=torch.long),
        "channel_ids": torch.arange(1, channels + 1, dtype=torch.int16),
    }


class Legacy360PatchAxialTests(unittest.TestCase):
    def make_model(self, base_fusion: str = "max") -> Legacy360PatchAxial:
        torch.manual_seed(19)
        return Legacy360PatchAxial(
            TinyPatchBackbone(),
            embed_dim=16,
            temporal_heads=4,
            sensor_heads=4,
            temporal_frame_blocks=0,
            base_fusion=base_fusion,
        )

    def test_zero_init_one_sensor_exactly_matches_legacy_concat(self):
        model = self.make_model()
        self.assertTrue(model.residual_is_zero)
        indices = torch.tensor([101, 707], dtype=torch.long)
        s2 = sensor_batch(
            global_rows=[101, 101, 101, 707, 707, 707],
            roles=[0, 1, 2, 0, 1, 2],
            values=[1.0, 2.0, 4.0, 3.0, 5.0, 8.0],
        )
        batch = {
            "index": indices,
            "labels": torch.tensor([0, 1]),
            "sensor_batches": OrderedDict(
                [
                    ("s2", s2),
                    ("l89", empty_sensor_batch()),
                    ("emit", empty_sensor_batch()),
                    ("s5p", empty_sensor_batch(channels=1)),
                ]
            ),
        }
        output = model(batch)

        # Reconstruct exactly what the historical ConcatTemporalDataset did.
        direct_images = torch.stack(
            (
                torch.cat((s2["images"][0], s2["images"][1], s2["images"][2]), dim=0),
                torch.cat((s2["images"][3], s2["images"][4], s2["images"][5]), dim=0),
            )
        )
        direct_ids = s2["channel_ids"].repeat(3).unsqueeze(0).expand(2, -1)
        cls = model.backbone.forward_features(
            {"imgs": direct_images, "chn_ids": direct_ids}
        )["x_norm_clstoken"]
        expected = _two_class_log_odds(model.heads["s2"](cls))
        torch.testing.assert_close(output.fused_logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            output.base_fused_logits, expected, rtol=0, atol=0
        )
        self.assertTrue(output.sensor_valid[:, 0].all())
        self.assertFalse(output.sensor_valid[:, 1:].any())
        self.assertEqual(
            int(torch.count_nonzero(output.temporal_residual_norm)), 0
        )

    def test_missing_sensor_mask_and_max_base_are_exact(self):
        model = self.make_model(base_fusion="max")
        batch = {
            "index": torch.tensor([11, 23]),
            "labels": torch.tensor([1, 0]),
            "sensor_batches": OrderedDict(
                [
                    (
                        "s2",
                        sensor_batch(
                            global_rows=[11, 11, 11, 23, 23, 23],
                            roles=[0, 1, 2, 0, 1, 2],
                            values=[1, 2, 3, 2, 3, 4],
                        ),
                    ),
                    (
                        "l89",
                        sensor_batch(
                            global_rows=[11, 11, 11],
                            roles=[0, 1, 2],
                            values=[5, 6, 7],
                        ),
                    ),
                    ("emit", empty_sensor_batch()),
                    ("s5p", empty_sensor_batch(channels=1)),
                ]
            ),
        }
        output = model(batch)
        expected = output.sensor_logits.masked_fill(
            ~output.sensor_valid, -torch.inf
        ).max(dim=1).values
        torch.testing.assert_close(
            output.fused_logits, expected, rtol=0, atol=0
        )
        self.assertTrue(output.sensor_valid[0, :2].all())
        self.assertTrue(output.sensor_valid[1, 0])
        self.assertFalse(output.sensor_valid[1, 1])
        self.assertEqual(float(output.sensor_attention[1, 1]), 0.0)
        torch.testing.assert_close(
            output.sensor_attention.sum(dim=1),
            torch.ones(2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_universal_row_head_preserves_historical_feature_max(self):
        model = self.make_model(base_fusion="max")
        torch.manual_seed(29)
        model.row_fusion_head = LegacyCLSHead(16)
        batch = {
            "index": torch.tensor([11]),
            "labels": torch.tensor([1]),
            "sensor_batches": OrderedDict(
                [
                    (
                        "s2",
                        sensor_batch(
                            global_rows=[11, 11, 11],
                            roles=[0, 1, 2],
                            values=[1, 2, 3],
                        ),
                    ),
                    (
                        "l89",
                        sensor_batch(
                            global_rows=[11, 11, 11],
                            roles=[0, 1, 2],
                            values=[5, 6, 7],
                        ),
                    ),
                    ("emit", empty_sensor_batch()),
                    ("s5p", empty_sensor_batch(channels=1)),
                ]
            ),
        }
        output = model(batch)
        row_positions = model._row_positions(batch["index"])
        s2_evidence, _, _ = model._one_sensor(
            "s2", batch["sensor_batches"]["s2"], row_positions=row_positions, batch_size=1
        )
        l89_evidence, _, _ = model._one_sensor(
            "l89", batch["sensor_batches"]["l89"], row_positions=row_positions, batch_size=1
        )
        expected_feature = torch.stack((s2_evidence, l89_evidence), dim=1).max(dim=1).values
        expected = _two_class_log_odds(model.row_fusion_head(expected_feature))
        torch.testing.assert_close(output.fused_logits, expected, rtol=0, atol=0)

    def test_first_step_opens_both_zero_residual_paths(self):
        model = self.make_model()
        batch = {
            "index": torch.tensor([31, 47]),
            "labels": torch.tensor([1, 0]),
            "sensor_batches": OrderedDict(
                [
                    (
                        "s2",
                        sensor_batch(
                            global_rows=[31, 31, 31, 47, 47, 47],
                            roles=[0, 1, 2, 0, 1, 2],
                            values=[1, 4, 8, 2, 3, 9],
                        ),
                    ),
                    ("l89", empty_sensor_batch()),
                    ("emit", empty_sensor_batch()),
                    ("s5p", empty_sensor_batch(channels=1)),
                ]
            ),
        }
        before = model(batch).fused_logits.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.fused_logits, batch["labels"].float()
        )
        optimizer.zero_grad()
        loss.backward()
        self.assertGreater(
            float(model.temporal["s2"].residual_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.sensor_fusion.output.weight.grad.abs().sum()), 0.0
        )
        optimizer.step()
        self.assertFalse(model.residual_is_zero)
        after = model(batch).fused_logits.detach()
        self.assertGreater(float((after - before).abs().max()), 0.0)

    def test_rejects_history_without_current(self):
        model = self.make_model()
        batch = {
            "index": torch.tensor([9]),
            "labels": torch.tensor([0]),
            "sensor_batches": OrderedDict(
                [
                    (
                        "s2",
                        sensor_batch(
                            global_rows=[9, 9],
                            roles=[1, 2],
                            values=[2, 3],
                        ),
                    ),
                    ("l89", empty_sensor_batch()),
                    ("emit", empty_sensor_batch()),
                    ("s5p", empty_sensor_batch(channels=1)),
                ]
            ),
        }
        with self.assertRaisesRegex(ValueError, "without a valid current sensor"):
            model(batch)

    def test_sealed_manifest_needs_explicit_authorization(self):
        with tempfile.TemporaryDirectory(prefix="legacy360_test_") as directory:
            path = Path(directory) / "manifest_test.csv"
            pd.DataFrame(
                [
                    {
                        "id": "one",
                        "plume_id": "event-A",
                        "label": 1,
                        "cluster_id": "cluster",
                        "macro_region_id": "region",
                        "s2_0_path": "/tmp/s2_0.tif",
                        "s2_90_path": "/tmp/s2_90.tif",
                        "s2_360_path": "/tmp/s2_360.tif",
                        "l89_0_path": "",
                        "l89_90_path": "",
                        "l89_360_path": "",
                        "emit_0_path": "",
                        "emit_90_path": "",
                        "emit_360_path": "",
                        "s5p_0_path": "",
                    }
                ]
            ).to_csv(path, index=False)
            with self.assertRaises(UnsafePathError):
                Query360Dataset(path)
            with self.assertRaises(PermissionError):
                AuthorizedSealedQuery360Dataset(path, authorization="")
            dataset = AuthorizedSealedQuery360Dataset(
                path, authorization=SEALED_TEST_AUTHORIZATION
            )
            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
