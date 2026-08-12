#!/usr/bin/env python3
"""CPU contract tests for the six-time two-axis model."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from experiments_tryout.six_time_multisensorfusion.model import (
    SensorNormalizationStats,
    SensorSixTimeInput,
    SixTimeMultiSensorTwoAxisModel,
    load_train_normalization_stats,
)


class FakePanopticonEncoder(nn.Module):
    """Small differentiable encoder with the same CLS output contract."""

    embed_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.embed_dim)
        self.forward_calls = 0
        self.observations = 0

    def forward_features(self, x_dict):
        images = x_dict["imgs"]
        channel_ids = x_dict["chn_ids"]
        self.forward_calls += 1
        self.observations += int(images.shape[0])
        image_scalar = images.mean(dim=(1, 2, 3))
        if channel_ids.ndim == 3:
            channel_ids = channel_ids[:, :, 0]
        wavelength_scalar = channel_ids.float().mean(dim=1) / 1000.0
        features = torch.stack((image_scalar, wavelength_scalar), dim=-1)
        return {"x_norm_clstoken": self.projection(features)}


def _sensor_input(
    valid: torch.Tensor,
    *,
    channels: int,
    seed: int,
    quality_dim: int = 2,
) -> SensorSixTimeInput:
    generator = torch.Generator().manual_seed(seed)
    batch_size = valid.shape[0]
    images = torch.randn(
        batch_size,
        6,
        channels,
        4,
        4,
        generator=generator,
    )
    channel_ids = torch.linspace(450.0, 2200.0, channels)
    gaps = torch.tensor([0.0, 7.0, 30.0, 90.0, 180.0, 365.0]).repeat(
        batch_size, 1
    )
    quality = torch.randn(
        batch_size,
        6,
        quality_dim,
        generator=generator,
    )
    duplicate = torch.zeros(batch_size, 6, dtype=torch.bool)
    duplicate[:, -1] = True
    return SensorSixTimeInput(
        images=images,
        channel_ids=channel_ids,
        time_valid=valid,
        time_delta_days=gaps,
        quality=quality,
        history_duplicate=duplicate,
    )


def synthetic_inputs() -> dict[str, SensorSixTimeInput]:
    s2_valid = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, False, False, False, False],
            [True, False, False, False, False, False],
        ]
    )
    emit_valid = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, True, True, True, True, True],  # orphan histories ignored
            [True, False, True, False, False, False],
        ]
    )
    return {
        "s2_cdse": _sensor_input(s2_valid, channels=3, seed=10),
        "emit": _sensor_input(emit_valid, channels=2, seed=20),
    }


def build_model(*, freeze_transient_encoder: bool = True):
    appearance = FakePanopticonEncoder()
    transient = FakePanopticonEncoder()
    model = SixTimeMultiSensorTwoAxisModel(
        appearance,
        transient,
        quality_dim=2,
        pair_hidden_dim=16,
        transient_dim=7,
        gate_hidden_dim=9,
        role_embedding_dim=4,
        sensor_embedding_dim=3,
        dropout=0.0,
        encode_chunk_size=4,
        normalization_stats={
            "s2_cdse": SensorNormalizationStats(
                mean=(0.1, 0.2, 0.3),
                std=(1.0, 2.0, 3.0),
            ),
            "emit": SensorNormalizationStats(
                mean=(0.4, 0.5),
                std=(1.5, 2.5),
            ),
        },
        freeze_appearance_encoder=True,
        freeze_transient_encoder=freeze_transient_encoder,
    )
    return model, appearance, transient


class TwoAxisModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_rejects_shared_axis_encoder(self):
        encoder = FakePanopticonEncoder()
        with self.assertRaisesRegex(ValueError, "different modules"):
            SixTimeMultiSensorTwoAxisModel(encoder, encoder, quality_dim=0)

    def test_axis_parameters_have_no_shared_storage(self):
        model, appearance, transient = build_model()
        del model
        a_storage = {
            parameter.untyped_storage().data_ptr()
            for parameter in appearance.parameters()
        }
        b_storage = {
            parameter.untyped_storage().data_ptr()
            for parameter in transient.parameters()
        }
        self.assertFalse(a_storage & b_storage)

    def test_shapes_masks_and_initial_exact_identity(self):
        model, appearance, transient = build_model()
        model.eval()
        output = model(synthetic_inputs(), return_features=True)

        self.assertEqual(tuple(output.fused_logits.shape), (3,))
        self.assertEqual(tuple(output.sensor_logits.shape), (3, 4))
        self.assertEqual(tuple(output.sensor_valid.shape), (3, 4))
        self.assertEqual(tuple(output.history_attention.shape), (3, 4, 5))
        self.assertEqual(tuple(output.effective_time_valid.shape), (3, 4, 6))
        self.assertEqual(tuple(output.appearance_features.shape), (3, 4, 8))
        self.assertEqual(tuple(output.transient_features.shape), (3, 4, 6, 8))
        self.assertTrue(bool(torch.isfinite(output.fused_logits).all()))

        # Zero-initialized B is exactly A, and fixed fusion is exactly A.
        self.assertTrue(
            torch.equal(
                output.transient_sensor_logits,
                output.appearance_sensor_logits,
            )
        )
        self.assertTrue(
            torch.equal(output.sensor_logits, output.appearance_sensor_logits)
        )
        self.assertTrue(
            torch.equal(output.fused_logits, output.appearance_fused_logits)
        )
        self.assertTrue(bool((output.residual_sensor_logits == 0).all()))

        # Only one encoder object is used for every valid B timepoint.
        expected_current = int(output.sensor_valid.sum())
        expected_all_times = int(output.effective_time_valid.sum())
        self.assertEqual(appearance.observations, expected_current)
        self.assertEqual(transient.observations, expected_all_times)
        self.assertGreater(transient.forward_calls, 1)  # chunking exercised

        history_mask = output.effective_time_valid[:, :, 1:]
        self.assertTrue(
            torch.equal(
                output.history_attention[~history_mask],
                torch.zeros_like(output.history_attention[~history_mask]),
            )
        )
        attention_sums = output.history_attention.sum(dim=-1)
        self.assertTrue(
            torch.allclose(
                attention_sums,
                output.history_valid.to(attention_sums.dtype),
            )
        )
        # Exactly one history means exact attention weight one.
        self.assertEqual(float(output.history_attention[1, 0, 0]), 1.0)
        self.assertEqual(float(output.history_attention[2, 2, 1]), 1.0)

        # Missing sensors/current times cannot leak classifier biases.
        self.assertTrue(bool((output.sensor_logits[:, 1] == 0).all()))
        self.assertEqual(float(output.sensor_logits[1, 2]), 0.0)

    def test_invalid_nan_inf_payloads_never_reach_encoder(self):
        model, _, _ = build_model()
        model.eval()
        inputs = synthetic_inputs()
        baseline = model(inputs)

        changed: dict[str, SensorSixTimeInput] = {}
        for sensor, item in inputs.items():
            images = item.images.clone()
            ids = item.channel_ids.reshape(1, 1, -1).expand(
                images.shape[0], 6, -1
            ).clone()
            invalid = ~item.time_valid
            images[invalid] = float("nan")
            ids[invalid] = float("inf")
            changed[sensor] = replace(item, images=images, channel_ids=ids)
        changed_output = model(changed)

        for field in (
            "fused_logits",
            "sensor_logits",
            "appearance_sensor_logits",
            "transient_sensor_logits",
            "history_attention",
        ):
            self.assertTrue(
                torch.equal(getattr(baseline, field), getattr(changed_output, field)),
                msg=field,
            )

    def test_no_history_forces_zero_residual_even_with_nonzero_bias(self):
        model, _, _ = build_model()
        with torch.no_grad():
            model.residual_bias.fill_(2.0)
            model.residual_weight.fill_(0.5)
        output = model(synthetic_inputs())
        # Row 2 S2 has no history; row 0 EMIT has no history.
        self.assertEqual(float(output.residual_sensor_logits[2, 0].detach()), 0.0)
        self.assertEqual(float(output.residual_sensor_logits[0, 2].detach()), 0.0)
        self.assertEqual(
            float(output.transient_sensor_logits[2, 0].detach()),
            float(output.appearance_sensor_logits[2, 0].detach()),
        )

    def test_every_row_requires_a_current_sensor(self):
        model, _, _ = build_model()
        inputs = synthetic_inputs()
        s2 = inputs["s2_cdse"]
        emit = inputs["emit"]
        s2_valid = s2.time_valid.clone()
        emit_valid = emit.time_valid.clone()
        s2_valid[1, 0] = False
        emit_valid[1, 0] = False
        inputs["s2_cdse"] = replace(s2, time_valid=s2_valid)
        inputs["emit"] = replace(emit, time_valid=emit_valid)
        with self.assertRaisesRegex(ValueError, "at least one valid current"):
            model(inputs)

    def test_fixed_axis_weights_are_not_parameters(self):
        model, _, _ = build_model()
        parameter_names = {name for name, _ in model.named_parameters()}
        buffer_names = {name for name, _ in model.named_buffers()}
        self.assertNotIn("axis_score_weights", parameter_names)
        self.assertIn("axis_score_weights", buffer_names)
        self.assertTrue(
            torch.equal(model.axis_score_weights, torch.tensor([0.5, 0.5]))
        )

    def test_appearance_only_mode_skips_transient_encoder(self):
        model, appearance, transient = build_model()
        output = model(
            synthetic_inputs(),
            return_features=True,
            compute_transient=False,
        )
        self.assertGreater(appearance.observations, 0)
        self.assertEqual(transient.observations, 0)
        self.assertIsNone(output.transient_features)
        self.assertTrue(
            torch.equal(output.sensor_logits, output.appearance_sensor_logits)
        )
        self.assertTrue(
            torch.equal(output.fused_logits, output.appearance_fused_logits)
        )

    def test_transient_stage_freezes_a_and_reaches_b_encoder_after_update(self):
        model, appearance, transient = build_model(freeze_transient_encoder=False)
        model.configure_training_stage("transient", train_axis_encoder=True)
        model.train()
        self.assertFalse(appearance.training)
        self.assertTrue(transient.training)
        self.assertFalse(any(parameter.requires_grad for parameter in appearance.parameters()))
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.appearance_heads.parameters())
        )

        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        inputs = synthetic_inputs()
        labels = torch.tensor([1.0, 0.0, 1.0])

        # Step 1 trains the zero-initialized residual readout.
        first = model(inputs)
        first_loss = nn.functional.binary_cross_entropy_with_logits(
            first.transient_fused_logits, labels
        )
        first_loss.backward()
        self.assertIsNotNone(model.residual_weight.grad)
        self.assertGreater(float(model.residual_weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in appearance.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Step 2 can now propagate through the nonzero residual readout into B.
        second = model(inputs)
        second_loss = nn.functional.binary_cross_entropy_with_logits(
            second.transient_fused_logits, labels
        )
        second_loss.backward()
        self.assertIsNotNone(transient.projection.weight.grad)
        self.assertGreater(float(transient.projection.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in appearance.parameters()))

    def test_state_dict_roundtrip_is_exact(self):
        first, _, _ = build_model()
        second, _, _ = build_model()
        second.load_state_dict(first.state_dict(), strict=True)
        first.eval()
        second.eval()
        inputs = synthetic_inputs()
        output_1 = first(inputs)
        output_2 = second(inputs)
        self.assertTrue(torch.equal(output_1.fused_logits, output_2.fused_logits))
        self.assertTrue(
            torch.equal(output_1.history_attention, output_2.history_attention)
        )

    def test_train_stats_loader_rejects_non_train_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(
                json.dumps(
                    {
                        "source_split": "train",
                        "sensors": {
                            "s2_cdse": {"mean": [1.0], "std": [2.0]}
                        },
                    }
                ),
                encoding="utf-8",
            )
            stats = load_train_normalization_stats(path)
            self.assertEqual(tuple(stats["s2_cdse"].mean), (1.0,))

            path.write_text(
                json.dumps(
                    {
                        "source_split": "all",
                        "sensors": {
                            "s2_cdse": {"mean": [1.0], "std": [2.0]}
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly 'train'"):
                load_train_normalization_stats(path)


if __name__ == "__main__":
    unittest.main()
