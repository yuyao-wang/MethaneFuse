#!/usr/bin/env python3
"""CPU contracts for the ragged legacy360 patch cache."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from legacy360_patch_cache import (  # noqa: E402
    SCHEMA_VERSION,
    CachedPatchAxialHead,
    PatchShardDataset,
    atomic_json,
    atomic_torch,
    deterministic_orthogonal_projection,
    projection_metadata,
    tensor_sha256,
    validate_shard,
)


def synthetic_shard() -> dict:
    torch.manual_seed(13)
    # obs0..2 = row0/s2/t0..2, obs3..4 = row0/l89/t0..1,
    # obs5..7 = row1/s2/t0..2.
    patch_counts = [3, 3, 3, 2, 2, 3, 3, 3]
    offsets = [0]
    for count in patch_counts:
        offsets.append(offsets[-1] + count)
    patches = torch.randn(offsets[-1], 64).half()
    valid = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    base_sensor_logits = torch.tensor(
        [[0.7, -0.2, 0.0, 0.0], [-1.1, 0.0, 0.0, 0.0]]
    )
    shard = {
        "schema_version": SCHEMA_VERSION,
        "row_index": torch.tensor([101, 909], dtype=torch.int64),
        "labels": torch.tensor([1, 0], dtype=torch.int64),
        "ids": ["a", "b"],
        "plume_ids": ["event-a", "event-b"],
        "availability_signatures": ["s2+l89", "s2"],
        "base_sensor_features_projected": torch.randn(2, 4, 64).half(),
        "base_sensor_valid": valid,
        "base_sensor_logits": base_sensor_logits,
        "base_fused_logits": torch.tensor([0.35, -0.8]),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "observation_row": torch.tensor(
            [0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32
        ),
        "observation_sensor": torch.tensor(
            [0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.int8
        ),
        "observation_role": torch.tensor(
            [0, 1, 2, 0, 1, 0, 1, 2], dtype=torch.int8
        ),
        "patch_layers": ["early", "final"],
        "patch_count_by_sensor": {
            "s2": [3, 3, 3, 3, 3, 3],
            "l89": [2, 2],
            "emit": [],
            "s5p": [],
        },
        "patches_early": patches,
        "patches_final": (patches.float() * 0.5).half(),
    }
    validate_shard(shard)
    return shard


class ProjectionTests(unittest.TestCase):
    def test_projection_is_deterministic_and_orthogonal(self):
        left = deterministic_orthogonal_projection(32, 8, seed=77)
        right = deterministic_orthogonal_projection(32, 8, seed=77)
        other = deterministic_orthogonal_projection(32, 8, seed=78)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(tensor_sha256(left), tensor_sha256(right))
        self.assertNotEqual(tensor_sha256(left), tensor_sha256(other))
        identity = left.T @ left
        torch.testing.assert_close(
            identity, torch.eye(8), rtol=1e-5, atol=1e-5
        )
        metadata = projection_metadata(left, seed=77)
        self.assertLess(metadata["max_abs_qtq_minus_i"], 1e-5)


class CachedHeadTests(unittest.TestCase):
    def test_epoch_zero_is_exact_cached_base(self):
        torch.manual_seed(31)
        model = CachedPatchAxialHead()
        shard = synthetic_shard()
        output = model(shard, patch_layer="early")
        self.assertTrue(model.residual_is_zero)
        torch.testing.assert_close(
            output.fused_logits,
            shard["base_fused_logits"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.sensor_logits,
            shard["base_sensor_logits"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            int(torch.count_nonzero(output.temporal_residual_norm)), 0
        )
        self.assertEqual(float(output.sensor_attention[1, 1]), 0.0)

    def test_first_step_reaches_temporal_projection_and_logit_residuals(self):
        torch.manual_seed(37)
        model = CachedPatchAxialHead()
        shard = synthetic_shard()
        output = model(shard, patch_layer="early")
        target = shard["labels"].float()
        main = torch.nn.functional.binary_cross_entropy_with_logits(
            output.fused_logits, target
        )
        expanded = target[:, None].expand_as(output.sensor_logits)
        per_sensor = torch.nn.functional.binary_cross_entropy_with_logits(
            output.sensor_logits, expanded, reduction="none"
        )
        mask = output.sensor_valid.float()
        loss = main + 0.2 * (per_sensor * mask).sum() / mask.sum()
        loss.backward()
        self.assertGreater(
            float(
                model.temporal["s2"]
                .residual_projection.weight.grad.abs()
                .sum()
            ),
            0.0,
        )
        self.assertGreater(
            float(model.sensor_fusion.output.weight.grad.abs().sum()), 0.0
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer.step()
        self.assertFalse(model.residual_is_zero)
        after = model(shard, patch_layer="early")
        self.assertGreater(
            float((after.fused_logits - output.fused_logits).abs().max()),
            0.0,
        )

    def test_ragged_unequal_roles_fail_closed(self):
        model = CachedPatchAxialHead()
        shard = synthetic_shard()
        # Make row0/s2/t1 two patches instead of three while retaining a valid
        # global ragged tensor.
        shard["offsets"][2:] -= 1
        shard["patches_early"] = shard["patches_early"][:-1]
        shard["patches_final"] = shard["patches_final"][:-1]
        validate_shard(shard)
        with self.assertRaisesRegex(ValueError, "unequal patch counts"):
            model(shard, patch_layer="early")

    def test_shard_roundtrip_dataset_never_needs_source_images(self):
        shard = synthetic_shard()
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            shard_dir = cache_dir / "shards"
            shard_dir.mkdir()
            shard_path = shard_dir / "shard_000000.pth"
            atomic_torch(shard_path, shard)
            atomic_json(
                cache_dir / "cache_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "partial": False,
                    "projection": {"output_dim": 64},
                    "checkpoint": {"checkpoint": "/not/read/source.pth"},
                    "patch_layers": ["early", "final"],
                    "early_blocks": 2,
                    "shards": [
                        {
                            "file": "shards/shard_000000.pth",
                            "rows": 2,
                            "observations": 8,
                            "patches_per_layer": int(shard["offsets"][-1]),
                            "bytes": shard_path.stat().st_size,
                        }
                    ],
                },
            )
            dataset = PatchShardDataset(cache_dir)
            loaded = dataset[0]
            torch.testing.assert_close(
                loaded["patches_early"],
                shard["patches_early"],
                rtol=0,
                atol=0,
            )
            self.assertEqual(loaded["ids"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
