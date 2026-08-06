#!/usr/bin/env python3

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from research.tempo_20260728 import (
    l89_supervised_lastblock_finetune as runner,
)


class TinyBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.projection(self.norm(tokens))


class TinyPanopticon(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(1, self.embed_dim)
        self.token_offset = nn.Parameter(
            torch.linspace(0.0, 0.07, self.embed_dim).view(1, 1, -1)
        )
        self.blocks = nn.ModuleList(
            [TinyBlock(self.embed_dim) for _ in range(12)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.prepare_calls = 0

    def prepare_tokens_with_masks(self, x_dict, masks=None):
        del masks
        self.prepare_calls += 1
        pooled = (
            x_dict["imgs"]
            .float()
            .mean(dim=(1, 2, 3), keepdim=False)
            .unsqueeze(-1)
        )
        cls = self.stem(pooled).unsqueeze(1)
        patch = cls + self.token_offset
        return torch.cat((cls, patch), dim=1)


class TinyFrameDataset(Dataset):
    def __init__(self, rows: int = 4):
        self.images = torch.randn(rows, 6, 2, 4, 4)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (
            torch.tensor(index),
            self.images[index],
            torch.ones(6, dtype=torch.bool),
            torch.ones(6),
            torch.ones(6, dtype=torch.int8),
        )


class CountingGuard:
    def __init__(self):
        self.phases = []

    def check(self, phase):
        self.phases.append(str(phase))


def metadata(batch: int = 4, roles: int = 6):
    unique = torch.ones(batch, roles, dtype=torch.bool)
    unique[1, 2] = False
    unique[2, 4:] = False
    delta = torch.tensor(
        [[0.0, -1.0, -3.0, -7.0, -90.0, -365.0]]
    ).expand(batch, -1)
    quality = torch.ones(batch, roles)
    role = torch.arange(roles)
    return unique, delta, quality, role


class EventLabelSamplerTests(unittest.TestCase):
    def test_sampler_is_deterministic_label_balanced_and_event_aware(self):
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
        events = ["a", "a", "a", "a", "b", "c", "d", "d"]
        first = runner.EventLabelSampler(
            labels, events, draws_per_epoch=101, seed=17
        )
        second = runner.EventLabelSampler(
            labels, events, draws_per_epoch=101, seed=17
        )
        first.set_epoch(3)
        second.set_epoch(3)
        self.assertTrue(torch.equal(first.order(), second.order()))
        self.assertEqual(first.order_sha256(), second.order_sha256())
        sampled_labels = labels[first.order()]
        counts = torch.bincount(sampled_labels, minlength=2)
        self.assertLessEqual(abs(int(counts[0]) - int(counts[1])), 1)
        self.assertEqual(first.audit()["mixed_label_events"], 2)
        first.set_epoch(4)
        self.assertFalse(torch.equal(first.order(), second.order()))

    def test_sampler_rejects_unbounded_draw_count(self):
        with self.assertRaisesRegex(ValueError, "draws_per_epoch"):
            runner.EventLabelSampler(
                [0, 1],
                ["a", "b"],
                draws_per_epoch=runner.MAX_DRAWS_PER_EPOCH + 1,
                seed=1,
            )


class MatchedPairTests(unittest.TestCase):
    def build_model(self) -> runner.MatchedLastBlockPair:
        torch.manual_seed(9)
        return runner.MatchedLastBlockPair(
            TinyPanopticon(),
            num_roles=6,
            t0_index=0,
            head_hidden_dim=12,
            temporal_dim=8,
            dropout=0.0,
            periods_days=(1, 7, 30, 365),
        )

    def test_epoch0_arms_are_exact_and_trunk_is_shared_no_grad(self):
        model = self.build_model()
        self.assertEqual(
            model.arm_state_sha256()["current_only"],
            model.arm_state_sha256()["temporal_d1"],
        )
        images = torch.randn(4, 6, 2, 4, 4)
        unique, delta, quality, role = metadata()
        model.eval()
        logits = model.forward_images(
            images,
            torch.tensor([490.0, 560.0]),
            unique,
            delta,
            quality,
            role,
            encoder_microbatch=24,
            device=torch.device("cpu"),
            amp_dtype="float32",
        )
        self.assertTrue(
            torch.equal(logits["current_only"], logits["temporal_d1"])
        )
        self.assertEqual(model.trunk.prepare_calls, 1)
        self.assertEqual(len(model.trunk.blocks), 11)
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.trunk.parameters())
        )
        trunk_tokens = model.encode_shared_trunk(
            images,
            torch.tensor([490.0, 560.0]),
            encoder_microbatch=24,
            device=torch.device("cpu"),
            amp_dtype="float32",
        )
        self.assertFalse(trunk_tokens.requires_grad)

    def test_last_blocks_and_final_norms_receive_gradients_only(self):
        model = self.build_model()
        images = torch.randn(4, 6, 2, 4, 4)
        unique, delta, quality, role = metadata()
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        model.train()
        logits = model.forward_images(
            images,
            torch.tensor([490.0, 560.0]),
            unique,
            delta,
            quality,
            role,
            encoder_microbatch=24,
            device=torch.device("cpu"),
            amp_dtype="float32",
        )
        loss = sum(
            F.binary_cross_entropy_with_logits(value, labels)
            for value in logits.values()
        )
        loss.backward()
        for arm_name in runner.ARM_NAMES:
            arm = getattr(model, arm_name)
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in arm.final_block.parameters()
                )
            )
            self.assertTrue(
                all(
                    parameter.grad is not None
                    for parameter in arm.final_norm.parameters()
                )
            )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.trunk.parameters()
            )
        )

    def test_one_cpu_optimizer_step_is_finite(self):
        model = self.build_model()
        groups, counts = runner.optimizer_groups(
            model, block_lr=3e-6, head_lr=5e-4, weight_decay=0.05
        )
        self.assertGreater(counts["block11_and_final_norm"], 0)
        self.assertGreater(counts["t0_and_tempo_heads"], 0)
        optimizer = torch.optim.AdamW(groups)
        images = torch.randn(6, 6, 2, 4, 4)
        unique, delta, quality, role = metadata(batch=6)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        model.train()
        logits = model.forward_images(
            images,
            torch.tensor([490.0, 560.0]),
            unique,
            delta,
            quality,
            role,
            encoder_microbatch=36,
            device=torch.device("cpu"),
            amp_dtype="float32",
        )
        loss = 0.5 * sum(
            F.binary_cross_entropy_with_logits(value, labels)
            for value in logits.values()
        )
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        model.eval()
        updated = model.forward_images(
            images,
            torch.tensor([490.0, 560.0]),
            unique,
            delta,
            quality,
            role,
            encoder_microbatch=36,
            device=torch.device("cpu"),
            amp_dtype="float32",
        )
        self.assertTrue(
            torch.isfinite(updated["current_only"]).all()
            and torch.isfinite(updated["temporal_d1"]).all()
        )

    def test_train_batches_and_dev_eval_execute_runtime_guard(self):
        model = self.build_model()
        groups, _counts = runner.optimizer_groups(
            model, block_lr=3e-6, head_lr=5e-4, weight_decay=0.05
        )
        optimizer = torch.optim.AdamW(groups)
        dataset = TinyFrameDataset(rows=4)
        loader = DataLoader(
            dataset, batch_size=2, sampler=SequentialSampler(dataset)
        )
        unique, delta, quality, role = metadata(batch=4)
        payload = {
            "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "ids": ["i0", "i1", "i2", "i3"],
            "plume_ids": ["p0", "p1", "p2", "p3"],
            "event_ids": ["e0", "e1", "e2", "e3"],
            "unique_mask": unique,
            "valid_mask": torch.ones(4, 6, dtype=torch.bool),
            "image_valid_mask": torch.ones(4, 6, dtype=torch.bool),
            "valid_fraction": quality,
            "delta_days": delta,
            "role_index": role,
        }
        train_guard = CountingGuard()
        record = runner.train_one_epoch(
            model,
            loader,
            payload,
            torch.tensor([490.0, 560.0]),
            optimizer,
            device=torch.device("cpu"),
            amp_dtype="float32",
            encoder_microbatch=12,
            grad_clip=1.0,
            runtime_guard=train_guard,
        )
        self.assertEqual(int(record["rows_seen"]), 4)
        self.assertEqual(
            train_guard.phases, ["training_batch", "training_batch"]
        )
        eval_guard = CountingGuard()
        metrics, probabilities, labels = runner.evaluate(
            model,
            loader,
            payload,
            torch.tensor([490.0, 560.0]),
            device=torch.device("cpu"),
            amp_dtype="float32",
            encoder_microbatch=12,
            runtime_guard=eval_guard,
        )
        self.assertEqual(eval_guard.phases, ["development_evaluation"])
        self.assertEqual(set(metrics), set(runner.ARM_NAMES))
        self.assertEqual(set(probabilities), set(runner.ARM_NAMES))
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])


class SafetyAndBudgetTests(unittest.TestCase):
    def test_runtime_guard_fails_closed_on_wall_clock(self):
        now = [100.0]
        guard = runner.RuntimeGuard(
            device=torch.device("cpu"),
            wall_limit_minutes=1.0,
            max_cuda_allocated_gib=28.0,
            clock=lambda: now[0],
        )
        now[0] += 59.0
        self.assertEqual(guard.check("under_limit")["check_count"], 1)
        now[0] += 2.0
        with self.assertRaisesRegex(RuntimeError, "Wall-time guard exceeded"):
            guard.check("over_limit")

    def test_runtime_guard_fails_closed_on_cuda_peak_without_using_gpu(self):
        reset_calls = []
        guard = runner.RuntimeGuard(
            device=torch.device("cuda:0"),
            wall_limit_minutes=43.0,
            max_cuda_allocated_gib=28.0,
            clock=lambda: 10.0,
            cuda_peak_reader=lambda _device: int(28.01 * 2**30),
            cuda_peak_resetter=lambda device: reset_calls.append(str(device)),
        )
        self.assertEqual(reset_calls, ["cuda:0"])
        with self.assertRaisesRegex(RuntimeError, "CUDA allocation guard exceeded"):
            guard.check("synthetic_cuda_peak")

    def test_development_guard_refuses_test_sealed_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("outer_test", "sealed_eval", "subject_holdout"):
                with self.assertRaises(ValueError):
                    runner.assert_development_path(
                        root / name / "data.csv", purpose="probe"
                    )

    def test_hard_budget_caps(self):
        base = dict(
            epochs=3,
            patience=1,
            draws_per_epoch=4096,
            batch_size=24,
            encoder_microbatch=96,
            block_lr=3e-6,
            head_lr=5e-4,
            weight_decay=0.05,
            grad_clip=1.0,
            wall_limit_minutes=43.0,
            max_cuda_allocated_gib=28.0,
        )
        runner.validate_training_budget(argparse.Namespace(**base))
        for key, value in (
            ("epochs", 4),
            ("patience", 2),
            ("draws_per_epoch", 4097),
            ("batch_size", 25),
            ("wall_limit_minutes", 45.01),
            ("max_cuda_allocated_gib", 28.01),
        ):
            changed = dict(base)
            changed[key] = value
            with self.assertRaises(ValueError):
                runner.validate_training_budget(
                    argparse.Namespace(**changed)
                )


if __name__ == "__main__":
    unittest.main()
