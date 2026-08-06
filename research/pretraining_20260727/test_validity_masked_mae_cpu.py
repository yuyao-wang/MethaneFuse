#!/usr/bin/env python3
"""Nine CPU-only regression tests for validity-masked Residual-MAE."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727.multisensor_residual_runner import (
    SENSOR_SPECS,
    STREAM_SUFFIXES,
    VALIDITY_OBJECTIVE_VERSION,
    MethaneResidualMAE,
    MultiSensorResidualModel,
    SensorSpec,
    TemporalResidualDataset,
    ValidityMaskedMethaneResidualMAE,
    atomic_torch_save,
    build_data_signature,
    build_encoder_signature,
    build_resume_signature,
    load_pretrained_encoder,
    load_three_frames,
    resize_validity_tensor,
    state_dict_fingerprint,
)


def assert_close(
    actual: torch.Tensor | float,
    expected: torch.Tensor | float,
    *,
    atol: float = 1e-7,
) -> None:
    actual_tensor = torch.as_tensor(actual).detach().cpu()
    expected_tensor = torch.as_tensor(expected).detach().cpu()
    if not torch.allclose(actual_tensor, expected_tensor, atol=atol, rtol=0):
        raise AssertionError(
            f"not close: actual={actual_tensor}, expected={expected_tensor}, "
            f"atol={atol}"
        )


def make_adapter(stream_channels: dict[str, int]) -> ValidityMaskedMethaneResidualMAE:
    return ValidityMaskedMethaneResidualMAE(
        sensor_channels=stream_channels,
        img_size=(2, 2),
        patch_size=(1, 1),
        embed_dim=8,
        depth=1,
        num_heads=2,
        decoder_embed_dim=8,
        decoder_depth=1,
        decoder_num_heads=2,
        mlp_ratio=2.0,
        fuse_freq=1,
        mask_ratio=0.5,
        is_pretrain=True,
        drop=0.0,
        temporal_fusion_mode="current_query",
    )


def test_1_native_validity_semantics() -> None:
    with tempfile.TemporaryDirectory(prefix="validity_native.") as directory:
        root = Path(directory)
        frames = np.ones((3, 12, 4, 4), dtype=np.float32)
        frames[0, 0, 0, 0] = 0.0
        frames[0, 0, 0, 1] = np.nan
        frames[0, 0, 0, 2] = np.inf
        frames[1, 0, 1, 0] = 0.0
        frames[2, 0, 1, 1] = np.nan
        paths = []
        for index in range(3):
            path = root / f"s2_{index}.tif"
            tifffile.imwrite(path, frames[index])
            paths.append(path)
        row = pd.Series(
            {
                "path_t0": str(paths[0]),
                "path_prev1": str(paths[1]),
                "path_seasonal": str(paths[2]),
            }
        )
        raw, valid = load_three_frames(row, SENSOR_SPECS["s2"], "ch4")
        expected = np.isfinite(raw) & (raw != 0)
        if not np.array_equal(valid, expected):
            raise AssertionError("TIFF finite/nonzero validity changed.")
        if valid[0, 0, 0, :3].any():
            raise AssertionError("TIFF zero/NaN/Inf must all be invalid.")
        if (valid[0] & valid[1])[0, 1, 0]:
            raise AssertionError("Recent residual did not intersect t0/history.")
        if (valid[0] & valid[2])[0, 1, 1]:
            raise AssertionError("Seasonal residual did not intersect t0/history.")

        s5p = np.ones((6, 4, 4), dtype=np.float32)
        s5p[0, 0, 0] = 0.0
        s5p[0, 0, 1] = np.nan
        s5p[1, 1, 0] = np.nan
        s5p[4, 1, 1] = np.nan
        npz_path = root / "s5p.npz"
        np.savez(npz_path, ch4=s5p)
        s5p_raw, s5p_valid = load_three_frames(
            pd.Series({"image_path": str(npz_path)}),
            SENSOR_SPECS["s5p"],
            "ch4",
        )
        if not s5p_valid[0, 0, 0, 0]:
            raise AssertionError("Finite S5P zero must remain valid.")
        if s5p_valid[0, 0, 0, 1]:
            raise AssertionError("Non-finite S5P value must be invalid.")
        expected_recent = s5p_valid[0] & s5p_valid[1]
        expected_seasonal = s5p_valid[0] & s5p_valid[2]
        if expected_recent[0, 1, 0] or expected_seasonal[0, 1, 1]:
            raise AssertionError("S5P residual intersections are incorrect.")
        if not np.array_equal(s5p_valid, np.isfinite(s5p_raw)):
            raise AssertionError("S5P validity must be finite-only.")


def test_2_geometry_alignment() -> None:
    with tempfile.TemporaryDirectory(prefix="validity_geometry.") as directory:
        root = Path(directory)
        arrays = [
            np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4),
            np.arange(21, 37, dtype=np.float32).reshape(1, 4, 4),
            np.arange(41, 57, dtype=np.float32).reshape(1, 4, 4),
        ]
        arrays[0][0, 0, 0] = 0.0
        paths = []
        for index, array in enumerate(arrays):
            path = root / f"toy_{index}.tif"
            tifffile.imwrite(path, array)
            paths.append(path)
        csv_path = root / "toy.csv"
        pd.DataFrame(
            [
                {
                    "path_t0": str(paths[0]),
                    "path_prev1": str(paths[1]),
                    "path_seasonal": str(paths[2]),
                    "label": 1,
                }
            ]
        ).to_csv(csv_path, index=False)
        spec = SensorSpec(
            "toy",
            "tiff",
            1,
            "path_t0",
            "path_prev1",
            "path_seasonal",
        )
        common = dict(
            csv_path=csv_path,
            spec=spec,
            stats={"mean": [0.0], "std": [1.0]},
            image_size=4,
            current_clip=100.0,
            residual_clip=100.0,
            s5p_data_key="ch4",
            resize_on_device=False,
            return_validity=True,
        )
        baseline, _ = TemporalResidualDataset(augment=False, **common)[0]
        with mock.patch.object(
            torch,
            "rand",
            side_effect=[torch.tensor(0.1), torch.tensor(0.9)],
        ):
            flipped, _ = TemporalResidualDataset(augment=True, **common)[0]
        for key in baseline["values"]:
            if not torch.equal(
                flipped["values"][key],
                torch.flip(baseline["values"][key], dims=(-1,)),
            ):
                raise AssertionError(f"Value flip misaligned for {key}.")
            if not torch.equal(
                flipped["validity"][key],
                torch.flip(baseline["validity"][key], dims=(-1,)),
            ):
                raise AssertionError(f"Validity flip misaligned for {key}.")

    validity_224 = torch.ones(1, 1, 224, 224)
    validity_224[0, 0, 0, 0] = 0.0
    downsampled = resize_validity_tensor(validity_224, (112, 112))
    assert_close(downsampled[0, 0, 0, 0], 0.75)
    if downsampled.min() < 0 or downsampled.max() > 1:
        raise AssertionError("Area-resized validity escaped [0,1].")
    small = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    expected_up = small.repeat_interleave(2, -2).repeat_interleave(2, -1)
    if not torch.equal(resize_validity_tensor(small, (4, 4)), expected_up):
        raise AssertionError("Validity upsample is not nearest-neighbour.")


def test_3_manual_loss_and_gradient() -> None:
    model = make_adapter({"toy_current": 1})
    prediction = torch.tensor(
        [[[[2.0], [1000.0], [-1000.0], [99.0]]]]
    ).reshape(1, 4, 1).requires_grad_()
    target = torch.zeros_like(prediction).detach().requires_grad_()
    loss, _ = model.forward_loss(
        {"toy_current": prediction},
        {"toy_current": target},
        {"toy_current": torch.ones(1, 4)},
        {"toy_current": torch.ones(1, dtype=torch.bool)},
        {"toy_current": torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])},
    )
    assert_close(loss, 4.0)
    loss.backward()
    assert_close(prediction.grad[0, 0, 0], 4.0)
    if torch.count_nonzero(prediction.grad[0, 1:]) != 0:
        raise AssertionError("Invalid prediction elements received gradient.")
    if torch.count_nonzero(target.grad[0, 1:]) != 0:
        raise AssertionError("Invalid decoder targets received gradient.")


def test_4_all_valid_regression() -> None:
    streams = {f"toy_{suffix}": 1 for suffix in STREAM_SUFFIXES}
    base = MethaneResidualMAE(
        sensor_channels=streams,
        img_size=(2, 2),
        patch_size=(1, 1),
        embed_dim=8,
        depth=1,
        num_heads=2,
        decoder_embed_dim=8,
        decoder_depth=1,
        decoder_num_heads=2,
        mlp_ratio=2.0,
        fuse_freq=1,
        mask_ratio=0.5,
        is_pretrain=True,
        drop=0.0,
        temporal_fusion_mode="current_query",
    )
    masked = make_adapter(streams)
    torch.manual_seed(41)
    predictions = {
        stream: torch.randn(3, 4, 8) for stream in streams
    }
    targets = {stream: torch.randn(3, 4, 8) for stream in streams}
    fixed_mask = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0]] * 3
    )
    masks = {stream: fixed_mask.clone() for stream in streams}
    present = {
        stream: torch.ones(3, dtype=torch.bool) for stream in streams
    }
    validity = {
        stream: torch.ones_like(predictions[stream]) for stream in streams
    }
    legacy_loss, _ = base.forward_loss(
        predictions,
        targets,
        masks,
        present,
    )
    new_loss, _ = masked.forward_loss(
        predictions,
        targets,
        masks,
        present,
        validity,
    )
    assert_close(new_loss, legacy_loss, atol=1e-7)


def test_5_invalid_channels_and_partial_patches() -> None:
    model = make_adapter(
        {"s2_current": 12, "l89_current": 10, "partial_current": 2}
    )
    s2_prediction = torch.ones(1, 1, 12)
    s2_prediction[..., 8:10] = 1000.0
    s2_validity = torch.ones_like(s2_prediction)
    s2_validity[..., 8:10] = 0.0
    l89_prediction = torch.ones(1, 1, 10)
    l89_prediction[..., 8] = 1000.0
    l89_validity = torch.ones_like(l89_prediction)
    l89_validity[..., 8] = 0.0
    partial_prediction = torch.tensor([[[2.0, 4.0]]])
    partial_validity = torch.tensor([[[1.0, 0.25]]])
    predictions = {
        "s2_current": s2_prediction,
        "l89_current": l89_prediction,
        "partial_current": partial_prediction,
    }
    targets = {key: torch.zeros_like(value) for key, value in predictions.items()}
    masks = {key: torch.ones(1, 1) for key in predictions}
    present = {key: torch.ones(1, dtype=torch.bool) for key in predictions}
    validity = {
        "s2_current": s2_validity,
        "l89_current": l89_validity,
        "partial_current": partial_validity,
    }
    loss, metrics = model.forward_loss(
        predictions,
        targets,
        masks,
        present,
        validity,
    )
    assert_close(metrics["loss_s2_current"], 1.0)
    assert_close(metrics["loss_l89_current"], 1.0)
    expected_partial = (4.0 * 1.0 + 16.0 * 0.25) / 1.25
    assert_close(metrics["loss_partial_current"], expected_partial)
    assert_close(loss, (1.0 + 1.0 + expected_partial) / 3.0)


def test_6_empty_stream_handling() -> None:
    model = make_adapter({"toy_current": 1, "toy_recent": 1})
    predictions = {
        "toy_current": torch.tensor([[[3.0]]]),
        "toy_recent": torch.tensor([[[1000.0]]]),
    }
    targets = {key: torch.zeros_like(value) for key, value in predictions.items()}
    masks = {key: torch.ones(1, 1) for key in predictions}
    present = {key: torch.ones(1, dtype=torch.bool) for key in predictions}
    validity = {
        "toy_current": torch.ones(1, 1, 1),
        "toy_recent": torch.zeros(1, 1, 1),
    }
    loss, metrics = model.forward_loss(
        predictions,
        targets,
        masks,
        present,
        validity,
    )
    assert_close(loss, 9.0)
    assert_close(metrics["empty_stream_toy_recent"], 1.0)
    assert_close(metrics["excluded_samples_toy_recent"], 1.0)
    assert_close(metrics["valid_streams"], 1.0)
    assert_close(metrics["empty_streams"], 1.0)

    try:
        model.forward_loss(
            predictions,
            targets,
            masks,
            present,
            {key: torch.zeros_like(value) for key, value in predictions.items()},
        )
    except ValueError as error:
        if "no valid masked elements" not in str(error):
            raise
    else:
        raise AssertionError("All-empty validity did not fail fast.")


def test_7_s5p_coarse_field_semantics() -> None:
    field = torch.zeros(1, 1, 224, 224)
    field[0, 0, 0, 1] = float("nan")
    native_validity = torch.isfinite(field).float()
    if native_validity[0, 0, 0, 0] != 1:
        raise AssertionError("Finite S5P zero lost validity.")
    coverage = resize_validity_tensor(native_validity, (112, 112))
    assert_close(coverage[0, 0, 0, 0], 0.75)

    model = make_adapter({"s5p_current": 1})
    prediction = torch.full((2, 4, 1), 3.0)
    target = torch.zeros_like(prediction)
    validity = torch.tensor(
        [
            [[1.0], [1.0], [1.0], [1.0]],
            [[1.0], [0.0], [0.0], [0.0]],
        ]
    )
    loss, metrics = model.forward_loss(
        {"s5p_current": prediction},
        {"s5p_current": target},
        {"s5p_current": torch.ones(2, 4)},
        {"s5p_current": torch.ones(2, dtype=torch.bool)},
        {"s5p_current": validity},
    )
    assert_close(loss, 9.0)
    assert_close(metrics["valid_samples_s5p_current"], 2.0)


def tiny_multisensor_kwargs(*, validity: bool) -> dict[str, object]:
    return {
        "sensors": ("s2", "s5p"),
        "sharing": "shared",
        "image_size": 28,
        "patch_size": 14,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "fuse_freq": 1,
        "dropout": 0.0,
        "mask_ratio": 0.5,
        "decoder_embed_dim": 16,
        "decoder_depth": 1,
        "decoder_num_heads": 4,
        "validity_masked_reconstruction": validity,
    }


def test_8_tiny_end_to_end_backward() -> None:
    torch.manual_seed(83)
    model = MultiSensorResidualModel(
        mode="pretrain",
        **tiny_multisensor_kwargs(validity=True),
    )
    losses = []
    input_tensors = []
    decoder_gradient_records = []
    for sensor in ("s2", "s5p"):
        channels = SENSOR_SPECS[sensor].channels
        validity = {}
        values = {}
        for suffix in STREAM_SUFFIXES:
            key = f"{sensor}_{suffix}"
            mask = torch.ones(2, channels, 28, 28)
            if sensor == "s2":
                mask[:, 8:10] = 0.0
            else:
                mask[:, :, :7, :7] = 0.0
            value = (torch.randn_like(mask) * mask).requires_grad_()
            values[key] = value
            validity[key] = mask
            input_tensors.append(value)
        output = model(sensor, values, validity_by_stream=validity)
        if not torch.isfinite(output["loss"]):
            raise AssertionError(f"Non-finite tiny {sensor} loss.")
        assert model.backbone is not None
        for key, prediction in output["predictions"].items():
            prediction.retain_grad()
            decoder_gradient_records.append(
                (
                    key,
                    prediction,
                    model.backbone.patchify(validity[key]),
                )
            )
        losses.append(output["loss"])
    torch.stack(losses).mean().backward()
    finite_parameter_grads = [
        gradient
        for parameter in model.parameters()
        if (gradient := parameter.grad) is not None
    ]
    if not finite_parameter_grads:
        raise AssertionError("Tiny MAE produced no parameter gradients.")
    if not all(torch.isfinite(gradient).all() for gradient in finite_parameter_grads):
        raise AssertionError("Tiny MAE produced non-finite parameter gradients.")
    if not all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in input_tensors
    ):
        raise AssertionError("Tiny MAE produced non-finite input gradients.")
    for key, prediction, patchified_validity in decoder_gradient_records:
        if prediction.grad is None:
            raise AssertionError(f"Missing retained decoder gradient for {key}.")
        invalid = patchified_validity == 0
        if not invalid.any():
            raise AssertionError(f"No invalid decoder target positions for {key}.")
        if torch.count_nonzero(prediction.grad[invalid]) != 0:
            raise AssertionError(
                f"Invalid decoder target positions received gradient for {key}."
            )


def signature_args(*, validity: bool, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        sharing="shared",
        image_size=28,
        patch_size=14,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        fuse_freq=1,
        dropout=0.0,
        validity_masked_reconstruction=validity,
        current_clip=8.0,
        residual_clip=5.0,
        s5p_data_key="ch4",
        mode=mode,
        mask_ratio=0.5,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        batch_size=2,
        learning_rate=3e-4,
        weight_decay=0.05,
        grad_clip=1.0,
        balanced_pos_weight=True,
        amp=False,
        augment=False,
        seed=17,
        max_val_batches=1,
    )


def test_9_signature_and_transfer_contract() -> None:
    sensors = ("s2", "s5p")
    with tempfile.TemporaryDirectory(prefix="validity_transfer.") as directory:
        root = Path(directory)
        csv_path = root / "manifest.csv"
        stats_path = root / "stats.json"
        csv_path.write_text("label\n0\n", encoding="utf-8")
        stats_path.write_text("{}\n", encoding="utf-8")
        csvs = {
            sensor: {"train": csv_path, "val": csv_path} for sensor in sensors
        }
        event_audit = {
            "fingerprint": "synthetic-zero-overlap",
            "global_train_val_overlap_after": 0,
        }
        old_args = signature_args(validity=False, mode="pretrain")
        masked_args = signature_args(validity=True, mode="pretrain")
        old_encoder = build_encoder_signature(old_args, sensors)
        masked_encoder = build_encoder_signature(masked_args, sensors)
        old_data = build_data_signature(
            old_args, sensors, csvs, stats_path, event_audit
        )
        masked_data = build_data_signature(
            masked_args, sensors, csvs, stats_path, event_audit
        )
        if "validity_adapter" in old_encoder:
            raise AssertionError("Legacy encoder signature changed.")
        if old_data["schema_version"] != 2 or (
            "validity_reconstruction" in old_data["representation"]
        ):
            raise AssertionError("Legacy data signature changed.")
        validity_contract = masked_data["representation"][
            "validity_reconstruction"
        ]
        if (
            validity_contract["objective_version"]
            != VALIDITY_OBJECTIVE_VERSION
        ):
            raise AssertionError("Masked objective version is absent.")
        if masked_encoder["validity_adapter"]["source_sha256"] != (
            validity_contract["adapter_source_sha256"]
        ):
            raise AssertionError("Adapter hashes disagree across signatures.")
        resume_signature = build_resume_signature(
            masked_args,
            sensors,
            masked_encoder,
            masked_data,
            rounds=1,
            loader_lengths={"s2": 1, "s5p": 1},
        )
        if (
            resume_signature["pretrain_model"]["reconstruction_objective"]
            != VALIDITY_OBJECTIVE_VERSION
        ):
            raise AssertionError("Resume signature lacks masked objective.")

        base = MethaneResidualMAE(
            sensor_channels={"toy_current": 1},
            img_size=(2, 2),
            patch_size=(1, 1),
            embed_dim=8,
            depth=1,
            num_heads=2,
            decoder_embed_dim=8,
            decoder_depth=1,
            decoder_num_heads=2,
            mlp_ratio=2.0,
            fuse_freq=1,
            mask_ratio=0.5,
            is_pretrain=True,
            drop=0.0,
            temporal_fusion_mode="current_query",
        )
        adapter = make_adapter({"toy_current": 1})
        if {
            key: tuple(value.shape) for key, value in base.state_dict().items()
        } != {
            key: tuple(value.shape)
            for key, value in adapter.state_dict().items()
        }:
            raise AssertionError("Adapter state dict is not parameter-compatible.")

        torch.manual_seed(101)
        old_pretrain = MultiSensorResidualModel(
            mode="pretrain",
            **tiny_multisensor_kwargs(validity=False),
        )
        torch.manual_seed(103)
        finetune = MultiSensorResidualModel(
            mode="supervised",
            **tiny_multisensor_kwargs(validity=True),
        )
        torch.manual_seed(103)
        scratch = MultiSensorResidualModel(
            mode="supervised",
            **tiny_multisensor_kwargs(validity=True),
        )
        initial_head_sha = state_dict_fingerprint(
            finetune.state_dict(), prefix="heads."
        )
        if initial_head_sha != state_dict_fingerprint(
            scratch.state_dict(), prefix="heads."
        ):
            raise AssertionError("Matched heads did not initialize equally.")

        old_path = root / "old_unmasked.pth"
        atomic_torch_save(
            {
                "mode": "pretrain",
                "encoder_signature": old_encoder,
                "data_signature": old_data,
                "model": old_pretrain.state_dict(),
            },
            old_path,
        )
        try:
            load_pretrained_encoder(
                finetune,
                old_path,
                expected_encoder_signature=masked_encoder,
                expected_data_signature=masked_data,
            )
        except ValueError as error:
            if "signature" not in str(error):
                raise
        else:
            raise AssertionError("Old unmasked checkpoint was accepted.")

        torch.manual_seed(107)
        masked_pretrain = MultiSensorResidualModel(
            mode="pretrain",
            **tiny_multisensor_kwargs(validity=True),
        )
        masked_path = root / "masked.pth"
        atomic_torch_save(
            {
                "mode": "pretrain",
                "encoder_signature": masked_encoder,
                "data_signature": masked_data,
                "model": masked_pretrain.state_dict(),
            },
            masked_path,
        )
        report = load_pretrained_encoder(
            finetune,
            masked_path,
            expected_encoder_signature=masked_encoder,
            expected_data_signature=masked_data,
        )
        if report["encoder_coverage"] != 1.0:
            raise AssertionError("Masked encoder transfer coverage is not 1.")
        if not report["excluded_decoder_keys"]:
            raise AssertionError("Masked transfer did not exclude decoder state.")
        if report["target_head_sha256_after"] != initial_head_sha:
            raise AssertionError("Masked transfer changed classification heads.")


TESTS = [
    test_1_native_validity_semantics,
    test_2_geometry_alignment,
    test_3_manual_loss_and_gradient,
    test_4_all_valid_regression,
    test_5_invalid_channels_and_partial_patches,
    test_6_empty_stream_handling,
    test_7_s5p_coarse_field_semantics,
    test_8_tiny_end_to_end_backward,
    test_9_signature_and_transfer_contract,
]


def main() -> None:
    results = []
    started = time.time()
    for test in TESTS:
        test_started = time.time()
        test()
        result = {
            "test": test.__name__,
            "status": "passed",
            "elapsed_seconds": time.time() - test_started,
        }
        results.append(result)
        print("[CPU test] " + json.dumps(result, sort_keys=True), flush=True)
    print(
        "[CPU test suite] "
        + json.dumps(
            {
                "status": "passed",
                "tests": len(results),
                "device": "cpu",
                "elapsed_seconds": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
