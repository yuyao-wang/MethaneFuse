#!/usr/bin/env python3
"""CPU contract tests for the bounded Sidecar-RCTP fallback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from research.pretraining_20260727 import l89_ragged_cls_experiment as base
from research.pretraining_20260727 import rctp_l89_continue_pretrain as continuation
from research.pretraining_20260727 import rctp_l89_sidecar_fallback as target


class SidecarContractTest(unittest.TestCase):
    def test_zero_init_is_bit_exact_base_plus_zero(self) -> None:
        torch.manual_seed(1)
        sidecar = target.ResponseLowRankSidecar(12, 5, rank=8)
        features = torch.randn(4, 6, 12)
        response = torch.randn(5)
        residual = sidecar(features, response)
        self.assertTrue(sidecar.exact_noop)
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        joined = target.concatenate_base_residual(
            features, residual, scale=0.1
        )
        self.assertTrue(torch.equal(joined[..., :12], features))
        self.assertEqual(int(torch.count_nonzero(joined[..., 12:])), 0)
        pretext = target.select_pretext_residual_channel(features, residual)
        self.assertEqual(int(torch.count_nonzero(pretext)), 0)
        valid = torch.ones(4, 6, dtype=torch.bool)
        delta_days = torch.zeros(4, 6)
        visit = torch.tensor([0, 1, 2, 3])
        assembled = target.screen.assemble_probe_batch(
            pretext,
            valid,
            delta_days,
            visit,
            pretext[torch.arange(4), visit],
            torch.zeros(4, 3, 12),
        )
        self.assertEqual(
            int(torch.count_nonzero(assembled["clean_features"])), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(assembled["delta_features"])), 0
        )

    def test_only_sidecar_learns_and_clean_anchor_has_gradient(self) -> None:
        torch.manual_seed(2)
        frozen_encoder = torch.nn.Linear(7, 12)
        frozen_encoder.requires_grad_(False)
        sidecar = target.ResponseLowRankSidecar(12, 5, rank=8)
        optimizer = torch.optim.AdamW(sidecar.parameters(), lr=1e-2)
        source = torch.randn(3, 6, 7)
        with torch.no_grad():
            features = frozen_encoder(source)
        mask = torch.ones(3, 6, dtype=torch.bool)
        response = torch.randn(5)
        desired = torch.randn_like(features)
        residual = sidecar(features, response)
        loss = (residual - desired).square().mean()
        loss = loss + 0.1 * target.clean_anchor_loss(residual, mask)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in sidecar.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen_encoder.parameters()))
        optimizer.step()
        self.assertGreater(
            int(torch.count_nonzero(sidecar(features, response))), 0
        )

    def test_p4_p5_initialization_and_render_set_are_matched(self) -> None:
        target.set_seed(20260728)
        p4 = target.ResponseLowRankSidecar(16, 9, rank=8)
        target.set_seed(20260728)
        p5 = target.ResponseLowRankSidecar(16, 9, rank=8)
        self.assertEqual(
            base.state_dict_sha256(p4.state_dict()),
            base.state_dict_sha256(p5.state_dict()),
        )
        variants = torch.arange(2 * 3 * 1 * 2 * 2).reshape(2, 3, 1, 2, 2)
        p4_order = target.pretrain_variant_order(
            variants, continuation.ARM_P4
        )
        p5_order = target.pretrain_variant_order(
            variants, continuation.ARM_P5
        )
        self.assertTrue(torch.equal(p4_order[:, 0], variants[:, 2]))
        self.assertTrue(torch.equal(p5_order[:, 0], variants[:, 0]))
        self.assertTrue(
            torch.equal(
                torch.sort(p4_order.flatten(1), dim=1).values,
                torch.sort(p5_order.flatten(1), dim=1).values,
            )
        )
        fixed_residual = torch.randn(2, 3, 16)
        direct_base = torch.randn_like(fixed_residual)
        shuffled_base = direct_base.flip(0)
        first = target.select_pretext_residual_channel(
            direct_base, fixed_residual
        )
        second = target.select_pretext_residual_channel(
            shuffled_base, fixed_residual
        )
        self.assertTrue(torch.equal(first, second))
        self.assertIs(first, fixed_residual)

    def test_derived_p0_shape_and_base_boundary(self) -> None:
        base_features = torch.randn(5, 6, 12).half()
        unique = torch.ones(5, 6, dtype=torch.bool)
        derived = target.derive_sidecar_features(
            base_features,
            unique,
            sidecar=None,
            response_metadata=None,
            residual_scale=0.1,
            batch_size=2,
        )
        self.assertEqual(tuple(derived.shape), (5, 6, 24))
        self.assertTrue(torch.equal(derived[..., :12], base_features))
        self.assertEqual(int(torch.count_nonzero(derived[..., 12:])), 0)

    def test_forbidden_development_paths_fail_closed(self) -> None:
        for name in ("test", "sealed", "holdout"):
            with self.assertRaises(ValueError):
                target.assert_development_path(
                    Path(f"/tmp/rctp_{name}/artifact.pt"), purpose="unit"
                )
        target.assert_development_path(
            Path("/tmp/rctp_sidecar/train.pt"), purpose="unit"
        )

    def test_p0_cache_is_valid_and_preserves_base_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_train.pt"
            output = root / "derived_train.pt"
            features = torch.randn(4, 6, 12).half()
            labels = torch.tensor([0, 1, 0, 1])
            valid = torch.ones(4, 6, dtype=torch.bool)
            timestamps = torch.arange(24, dtype=torch.int64).reshape(4, 6)
            input_contract = {
                "script_version": "unit",
                "weights_sha256": "base-sha",
                "path_columns": [f"path_{i}" for i in range(6)],
                "time_columns": [f"time_{i}" for i in range(6)],
                "role_names": [f"role_{i}" for i in range(6)],
                "channel_ids": [1.0, 2.0],
                "csv_sha256": "csv",
                "input_table_sha256": "table",
                "source_rows": 4,
                "selected_rows": 4,
                "row_selection": "all",
                "row_selection_seed": 1,
            }
            payload = {
                "format_version": base.CACHE_FORMAT_VERSION,
                "script_version": "unit",
                "split": "train",
                "features": features,
                "labels": labels,
                "ids": [f"id-{i}" for i in range(4)],
                "plume_ids": [f"plume-{i}" for i in range(4)],
                "event_ids": [f"event-{i}" for i in range(4)],
                "timestamps_utc_ns": timestamps,
                "delta_days": torch.zeros(4, 6),
                "role_names": input_contract["role_names"],
                "role_index": torch.arange(6),
                "t0_index": 0,
                "valid_mask": valid,
                "duplicate_mask": torch.zeros_like(valid),
                "unique_mask": valid,
                "path_columns": input_contract["path_columns"],
                "time_columns": input_contract["time_columns"],
                "input_contract": input_contract,
                "input_contract_sha256": base.sha256_bytes(
                    base.canonical_json_bytes(input_contract)
                ),
                "csv_sha256": "csv",
                "weights_path": "/tmp/base.pth",
                "weights_sha256": "base-sha",
                "feature_sha256": base.tensor_sha256(features),
            }
            base.atomic_torch_save(source, payload)
            target.command_build_cache(
                SimpleNamespace(
                    arm="p0",
                    split="train",
                    base_cache=str(source),
                    sidecar_checkpoint=None,
                    output_cache=str(output),
                    batch_size=2,
                )
            )
            derived = base.torch_load_trusted(output)
            base.validate_cache_payload(
                derived, path=output, expected_split="train"
            )
            self.assertTrue(torch.equal(derived["features"][..., :12], features))
            self.assertEqual(
                int(torch.count_nonzero(derived["features"][..., 12:])), 0
            )


if __name__ == "__main__":
    unittest.main()
