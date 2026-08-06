"""Training/runtime helpers shared across MethaneFuse entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn

from thirdparty.dinov2.models.vision_transformer import DinoVisionTransformer

def _load_backbone(weights_path: Optional[str] = None, *, strict: bool = True) -> DinoVisionTransformer:
    from src.backbones import build_panopticon_vitb14

    backbone = build_panopticon_vitb14()
    if weights_path in (None, "", "none", "scratch", "random"):
        return backbone
    ckpt_path = Path(weights_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, Mapping) and "backbone" in state:
        state = state["backbone"]
    backbone.load_state_dict(state, strict=strict)
    return backbone


def _index_batch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, idx)


def _slice_x_dict(x_dict: MutableMapping[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: _index_batch(v, idx) for k, v in x_dict.items() if isinstance(v, torch.Tensor)}


def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [recursive_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

def load_model_checkpoint_flexible(path: Path, model: nn.Module, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Load the container on CPU: training checkpoints also contain optimizer
    # state, which is not used here and can otherwise exhaust a nearly-full GPU
    # before the model state is copied into the already-placed module.
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, Mapping) and "model" in ckpt else ckpt
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")

    model_keys = set(model.state_dict().keys())
    mapped_state: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped_key = str(key)
        if ".qkv." in mapped_key:
            candidate = mapped_key.replace(".qkv.", ".qkv.base_qkv.")
            if candidate in model_keys:
                mapped_key = candidate
        mapped_state[mapped_key] = value

    incompatible = model.load_state_dict(mapped_state, strict=False)
    missing = [key for key in incompatible.missing_keys if "q_experts" not in key and "v_experts" not in key and ".gate." not in key]
    if missing:
        print(f"[Checkpoint][Warn] Missing non-adapter keys while loading {path}: {missing[:20]}", flush=True)
    if incompatible.unexpected_keys:
        print(
            f"[Checkpoint][Warn] Unexpected keys while loading {path}: {incompatible.unexpected_keys[:20]}",
            flush=True,
        )
    print(f"Loaded model weights from {path}", flush=True)
