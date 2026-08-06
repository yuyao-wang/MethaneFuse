#!/usr/bin/env python3
"""CPU-only regression tests for the bounded L89 RCTP transfer loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import audit_rctp_l89_real_cls_loop as audit
from research.pretraining_20260727 import l89_ragged_cls_experiment as cache_runner
from research.pretraining_20260727 import rctp_l89_continue_pretrain as runner
from research.pretraining_20260727 import rctp_l89_screen as screen


class TinyBackbone(nn.Module):
    def __init__(self, channels: int = 2, dim: int = 8, depth: int = 4):
        super().__init__()
        self.embed_dim = dim
        self.input_projection = nn.Linear(channels, dim)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, inputs):
        images = inputs["imgs"]
        values = images.mean(dim=(-2, -1))
        output = self.input_projection(values)
        for block in self.blocks:
            output = output + block(output)
        return {"x_norm_clstoken": self.norm(output)}


class TinyOnlineDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        clean = 0.2 + 0.05 * torch.rand(
            3, 2, 8, 8, generator=generator
        )
        return {
            "row_index": torch.tensor(index, dtype=torch.long),
            "visit_index": torch.tensor(index % 3, dtype=torch.long),
            "peak_drop": torch.tensor(0.04, dtype=torch.float32),
            "clean_images": clean,
            "selected_valid_pixels": torch.ones(2, 8, 8, dtype=torch.bool),
            "plume_field": torch.ones(8, 8, dtype=torch.float32),
        }


class RCTPL89ContinuePretrainTests(unittest.TestCase):
    def test_final_block_freeze_contract_and_gradient(self) -> None:
        backbone = TinyBackbone(depth=4)
        audit = runner.configure_last_blocks(backbone, train_last_blocks=2)
        self.assertEqual(audit["trainable_block_indices"], [2, 3])
        trainable = {
            name
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith(("blocks.2.", "blocks.3.")) for name in trainable)
        )

        clean = torch.randn(2, 3, 2, 8, 8)
        variants = torch.randn(2, 3, 2, 8, 8)
        clean_cls, variant_cls = runner.encode_online_clean_and_variants(
            backbone, clean, variants, torch.tensor([1000.0, 2000.0])
        )
        self.assertEqual(tuple(clean_cls.shape), (2, 3, 8))
        self.assertEqual(tuple(variant_cls.shape), (2, 3, 8))
        anchor = runner.validity_masked_clean_anchor_loss(
            clean_cls,
            torch.randn_like(clean_cls),
            torch.ones(2, 3, dtype=torch.bool),
            metric="cosine",
        )
        anchor.backward()
        for index, block in enumerate(backbone.blocks):
            gradients = [parameter.grad for parameter in block.parameters()]
            if index < 2:
                self.assertTrue(all(value is None for value in gradients))
            else:
                self.assertTrue(any(value is not None for value in gradients))

    def test_zero_anchor_is_exact_original_loss_path(self) -> None:
        pretext = torch.tensor(2.5, requires_grad=True)
        online_clean = torch.randn(2, 3, 4, requires_grad=True)
        total, anchor = runner.combine_pretext_and_clean_anchor(
            pretext,
            online_clean,
            None,
            torch.ones(2, 3, dtype=torch.bool),
            weight=0.0,
            metric="cosine",
        )
        self.assertIs(total, pretext)
        self.assertEqual(float(anchor), 0.0)
        total.backward()
        self.assertEqual(float(pretext.grad), 1.0)
        self.assertIsNone(online_clean.grad)

    def test_anchor_is_clean_only_validity_masked_and_target_detached(self) -> None:
        torch.manual_seed(13)
        online_clean = torch.randn(2, 3, 5, requires_grad=True)
        base_target = torch.randn(2, 3, 5, requires_grad=True)
        variants = torch.randn(2, 3, 5, requires_grad=True)
        valid = torch.tensor([[True, False, True], [False, True, False]])
        loss = runner.validity_masked_clean_anchor_loss(
            online_clean, base_target, valid, metric="l2"
        )
        loss.backward()
        self.assertIsNone(base_target.grad)
        self.assertIsNone(variants.grad)
        self.assertGreater(float(online_clean.grad[valid].abs().sum()), 0.0)
        self.assertEqual(float(online_clean.grad[~valid].abs().sum()), 0.0)

    def test_p4_p5_use_same_variants_with_only_positive_order_changed(self) -> None:
        variants = torch.arange(3.0).view(1, 3, 1, 1, 1)
        p5 = runner.reorder_objective_variants(variants, runner.ARM_P5)
        p4 = runner.reorder_objective_variants(variants, runner.ARM_P4)
        self.assertEqual(p5.flatten().tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(p4.flatten().tolist(), [2.0, 1.0, 0.0])
        self.assertEqual(
            sorted(p5.flatten().tolist()), sorted(p4.flatten().tolist())
        )
        renderer = screen.RendererConfig(
            response=(0.0, 1.0),
            wavelength_shuffle=(1, 0),
        )
        self.assertEqual(
            runner.objective_response(renderer, runner.ARM_P5), (0.0, 1.0)
        )
        self.assertEqual(
            runner.objective_response(renderer, runner.ARM_P4), (1.0, 0.0)
        )

    def test_one_online_cpu_optimizer_step_without_cached_features(self) -> None:
        torch.manual_seed(7)
        backbone = TinyBackbone(depth=3)
        audit = runner.configure_last_blocks(backbone, train_last_blocks=1)
        renderer = screen.RendererConfig(
            response=(0.0, 1.0),
            wavelength_shuffle=(1, 0),
            min_peak_drop=0.04,
            max_peak_drop=0.04,
        )
        metadata = screen.response_metadata(
            runner.objective_response(renderer, runner.ARM_P5),
            (1000.0, 2000.0),
            timepoints=3,
        )
        probe = screen.RCTPTemporalProbe(
            feature_dim=8,
            response_dim=len(metadata),
            num_roles=3,
            model_dim=12,
            num_heads=3,
            dropout=0.0,
            periods_days=(1.0, 7.0),
        )
        groups, _ = runner.optimizer_parameter_groups(
            backbone,
            probe,
            backbone_lr=1e-3,
            probe_lr=1e-3,
            weight_decay=0.0,
        )
        optimizer = torch.optim.AdamW(groups)
        payload = {
            # Deliberately no "features": stale CLS use must not be required.
            "unique_mask": torch.ones(2, 3, dtype=torch.bool),
            "delta_days": torch.tensor(
                [[0.0, -5.0, -20.0], [0.0, -7.0, -30.0]]
            ),
            "role_index": torch.arange(3),
            "input_contract": {
                "normalization_mean": [0.0, 0.0],
                "normalization_std": [1.0, 1.0],
                "band_indices": [0, 1],
                "channel_ids": [1000.0, 2000.0],
            },
        }
        metrics, step, predictions = runner.run_epoch(
            backbone=backbone,
            trainable_block_indices=audit["trainable_block_indices"],
            probe=probe,
            loader=DataLoader(TinyOnlineDataset(), batch_size=2),
            payload=payload,
            base_clean_anchor_features=None,
            clean_anchor_weight=0.0,
            clean_anchor_metric="cosine",
            renderer=renderer,
            response_metadata=metadata,
            arm=runner.ARM_P5,
            device=torch.device("cpu"),
            amp_dtype="float32",
            optimizer=optimizer,
            scheduler=None,
            global_step=0,
            max_train_steps=1,
            loss_weights=(1.0, 0.25, 0.1),
            grad_clip=1.0,
            log_interval=100,
        )
        self.assertEqual(step, 1)
        self.assertEqual(len(predictions), 6)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])))
        self.assertEqual(metrics["matched_groups"], 2)
        self.assertEqual(metrics["clean_anchor_weight"], 0.0)
        self.assertEqual(metrics["clean_anchor_loss"], 0.0)
        self.assertAlmostEqual(
            metrics["total_loss"], metrics["pretext_total_loss"]
        )
        anchor_targets = torch.randn(2, 3, 8)
        anchored_metrics, anchored_step, _ = runner.run_epoch(
            backbone=backbone,
            trainable_block_indices=audit["trainable_block_indices"],
            probe=probe,
            loader=DataLoader(TinyOnlineDataset(), batch_size=2),
            payload=payload,
            base_clean_anchor_features=anchor_targets,
            clean_anchor_weight=0.1,
            clean_anchor_metric="cosine",
            renderer=renderer,
            response_metadata=metadata,
            arm=runner.ARM_P5,
            device=torch.device("cpu"),
            amp_dtype="float32",
            optimizer=optimizer,
            scheduler=None,
            global_step=step,
            max_train_steps=2,
            loss_weights=(1.0, 0.25, 0.1),
            grad_clip=1.0,
            log_interval=100,
        )
        self.assertEqual(anchored_step, 2)
        self.assertGreater(anchored_metrics["clean_anchor_loss"], 0.0)
        self.assertAlmostEqual(
            anchored_metrics["clean_anchor_weighted_loss"],
            0.1 * anchored_metrics["clean_anchor_loss"],
            places=6,
        )

    def test_transfer_checkpoint_has_load_backbone_contract(self) -> None:
        backbone = TinyBackbone()
        probe = nn.Linear(8, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dev_transfer.pth"
            runner.save_transfer_checkpoint(
                path,
                backbone=backbone,
                probe_state=probe.state_dict(),
                metadata={"arm": runner.ARM_P5},
            )
            payload = torch.load(path, map_location="cpu")
        self.assertEqual(
            set(payload), {"backbone", "rctp_probe_state", "rctp_metadata"}
        )
        clone = TinyBackbone()
        clone.load_state_dict(payload["backbone"], strict=True)
        self.assertEqual(payload["rctp_metadata"]["arm"], runner.ARM_P5)

    def test_event_balanced_threshold_does_not_let_large_event_dominate(self) -> None:
        event_ids = ["large"] * 4 + ["small"]
        weights = cache_runner.event_balanced_row_weights(event_ids)
        self.assertAlmostEqual(float(weights[:4].sum()), 1.0)
        self.assertAlmostEqual(float(weights[4:].sum()), 1.0)
        threshold, score = cache_runner.best_weighted_positive_f1_threshold(
            labels=torch.tensor([1, 0, 0, 0, 1]).numpy(),
            probabilities=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.55]).numpy(),
            sample_weights=weights,
        )
        self.assertAlmostEqual(threshold, 0.55)
        self.assertGreater(score, 0.75)

    def test_paired_event_bootstrap_uses_identical_event_draws(self) -> None:
        base = pd.DataFrame(
            {
                "id": [str(value) for value in range(8)],
                "plume_id": ["a"] * 4 + ["b"] * 4,
                "event_id": ["event-a"] * 4 + ["event-b"] * 4,
                "label": [1, 1, 0, 0, 1, 1, 0, 0],
            }
        )
        probabilities = {
            "p0": [0.6, 0.4, 0.7, 0.2, 0.6, 0.4, 0.7, 0.2],
            "p4": [0.7, 0.5, 0.6, 0.2, 0.7, 0.5, 0.6, 0.2],
            "p5": [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1],
        }
        arms = {}
        points = {}
        for arm, values in probabilities.items():
            frame = base.copy()
            frame["probability"] = values
            arms[arm] = {"predictions": frame}
            points[arm] = audit.compute_point_metrics(frame)
        intervals, deltas = audit.bootstrap_event_f1(
            arms, points, replicates=50, seed=9
        )
        self.assertEqual(
            intervals["p5"]["dev_selected"]["positive_f1"]["point"], 1.0
        )
        self.assertGreater(
            deltas["p5_minus_p0"]["fixed_0p5"]["positive_f1"]["point"], 0
        )


if __name__ == "__main__":
    unittest.main()
