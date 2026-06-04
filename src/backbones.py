"""Backbone builders used by MethaneFuse and its baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch

from thirdparty.dinov2.models import vision_transformer as vits


def build_panopticon_vitb14():
    """Build the inherited Panopticon ViT-B/14 backbone architecture."""

    return vits.vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0e-05,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        embed_layer="PanopticonPE",
        pe_args={
            "attn_dim": 2304,
            "chnfus_cfg": {
                "layer_norm": False,
                "attn_cfg": {"num_heads": 16},
            },
        },
    )


def load_panopticon_vitb14(weights_path: Optional[str] = None, *, strict: bool = True):
    """Build the inherited Panopticon backbone and optionally load local weights."""

    backbone = build_panopticon_vitb14()
    if weights_path in (None, "", "none", "scratch", "random"):
        return backbone

    checkpoint = torch.load(Path(weights_path), map_location="cpu")
    if isinstance(checkpoint, Mapping) and "backbone" in checkpoint:
        checkpoint = checkpoint["backbone"]
    backbone.load_state_dict(checkpoint, strict=strict)
    return backbone
