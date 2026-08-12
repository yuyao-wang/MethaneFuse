#!/usr/bin/env python3
"""Run a tiny end-to-end forward with two real Panopticon ViT-B encoders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Keep direct script execution equivalent to ``python -m ...``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_tryout.six_time_multisensorfusion.model import (
    SensorSixTimeInput,
    build_two_axis_panopticon_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/panopticon_vitb14_teacher.pth"),
    )
    args = parser.parse_args()

    model = build_two_axis_panopticon_model(
        args.weights,
        quality_dim=0,
        pair_hidden_dim=16,
        transient_dim=8,
        gate_hidden_dim=8,
        role_embedding_dim=4,
        sensor_embedding_dim=4,
        dropout=0.0,
        encode_chunk_size=1,
        freeze_appearance_encoder=True,
        freeze_transient_encoder=True,
    ).eval()

    images = torch.randn(1, 6, 3, 28, 28)
    valid = torch.tensor([[True, True, False, False, False, False]])
    inputs = {
        "s2_cdse": SensorSixTimeInput(
            images=images,
            channel_ids=torch.tensor([492.0, 664.0, 1613.0]),
            time_valid=valid,
            time_delta_days=torch.tensor([[0.0, 30.0, 0.0, 0.0, 0.0, 0.0]]),
        )
    }
    with torch.inference_mode():
        output = model(inputs, return_features=True)

    assert output.fused_logits.shape == (1,)
    assert output.sensor_logits.shape == (1, 4)
    assert output.appearance_features is not None
    assert output.appearance_features.shape == (1, 4, 768)
    assert output.transient_features is not None
    assert output.transient_features.shape == (1, 4, 6, 768)
    assert torch.equal(
        output.transient_sensor_logits,
        output.appearance_sensor_logits,
    )
    assert torch.equal(output.sensor_logits, output.appearance_sensor_logits)
    assert torch.isfinite(output.fused_logits).all()
    print(
        "real Panopticon smoke OK:",
        {
            "fused_logits": tuple(output.fused_logits.shape),
            "sensor_logits": tuple(output.sensor_logits.shape),
            "appearance_features": tuple(output.appearance_features.shape),
            "transient_features": tuple(output.transient_features.shape),
        },
    )


if __name__ == "__main__":
    main()
