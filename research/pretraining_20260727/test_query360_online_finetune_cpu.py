#!/usr/bin/env python3

import unittest

import torch
import torch.nn as nn

from research.pretraining_20260727.query360_online_finetune import (
    ARMS,
    EpochOrderSampler,
    OnlineTransientQueryClassifier,
)


class FakeBackbone(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, self.embed_dim)

    def forward_features(self, x_dict):
        pooled = x_dict["imgs"].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return {"x_norm_clstoken": self.projection(pooled)}


def synthetic_batch():
    valid = torch.zeros(2, 4, 3, dtype=torch.bool)
    valid[0, 0] = True
    valid[1, 0] = True
    return {
        "index": torch.tensor([101, 205]),
        "valid_mask": valid,
        "sensor_batches": {
            "s2": {
                "images": torch.randn(6, 1, 2, 2),
                "rows": torch.tensor([101, 101, 101, 205, 205, 205]),
                "roles": torch.tensor([0, 1, 2, 0, 1, 2]),
                "channel_ids": torch.tensor([490]),
            },
            "l89": {
                "images": torch.empty(0, 1, 0, 0),
                "rows": torch.empty(0, dtype=torch.long),
                "roles": torch.empty(0, dtype=torch.long),
                "channel_ids": torch.tensor([490]),
            },
            "emit": {
                "images": torch.empty(0, 1, 0, 0),
                "rows": torch.empty(0, dtype=torch.long),
                "roles": torch.empty(0, dtype=torch.long),
                "channel_ids": torch.tensor([490]),
            },
            "s5p": {
                "images": torch.empty(0, 1, 0, 0),
                "rows": torch.empty(0, dtype=torch.long),
                "roles": torch.empty(0, dtype=torch.long),
                "channel_ids": torch.tensor([0]),
            },
        },
    }


class OnlineFineTuneTests(unittest.TestCase):
    def test_online_runner_exposes_scale_aware_arm(self):
        self.assertEqual(
            ARMS,
            (
                "current_only",
                "transient_query",
                "scale_aware_transient_query",
            ),
        )

    def test_differentiable_scatter_reaches_backbone(self):
        model = OnlineTransientQueryClassifier(
            FakeBackbone(),
            model_dim=16,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        features, valid, observations = model.encode_batch(
            synthetic_batch(),
            device=torch.device("cpu"),
            amp_dtype="float32",
            encoder_microbatch=2,
        )
        self.assertEqual(tuple(features.shape), (2, 4, 3, 8))
        self.assertEqual(observations, 6)
        self.assertEqual(int(valid.sum()), 6)
        features.sum().backward()
        gradient = model.backbone.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_sampler_is_epoch_deterministic(self):
        first = EpochOrderSampler(100, seed=17, shuffle=True)
        second = EpochOrderSampler(100, seed=17, shuffle=True)
        for epoch in (1, 2, 9):
            first.set_epoch(epoch)
            second.set_epoch(epoch)
            self.assertTrue(torch.equal(first.order(), second.order()))
            self.assertEqual(first.order_sha256(), second.order_sha256())
        first.set_epoch(1)
        second.set_epoch(2)
        self.assertFalse(torch.equal(first.order(), second.order()))


if __name__ == "__main__":
    unittest.main()
