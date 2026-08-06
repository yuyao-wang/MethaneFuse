"""Shared normalized temporal-input transformations for sensor experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch


INPUT_MODES = ("raw", "current", "residual", "current_residual")


def parse_residual_slots(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    slots = tuple(str(item).strip() for item in values if str(item).strip())
    if not slots:
        raise ValueError("At least one residual slot is required")
    return slots


def output_timepoints(
    input_mode: str, source_timepoints: int, residual_slots: Sequence[str]
) -> int:
    if input_mode == "raw":
        return int(source_timepoints)
    if input_mode == "current":
        return 1
    if input_mode == "residual":
        return len(tuple(residual_slots))
    if input_mode == "current_residual":
        return 1 + len(tuple(residual_slots))
    raise ValueError(f"Unsupported input mode: {input_mode}")


def transform_temporal_sample(
    x_dict: dict[str, torch.Tensor],
    *,
    path_columns: Sequence[str],
    input_mode: str,
    residual_slots: Sequence[str],
    residual_clip: float | None,
) -> dict[str, torch.Tensor]:
    """Transform a normalized `(T,C,H,W)` sample into raw/current/residual streams.

    Residuals are computed after per-band normalization.  Channel IDs remain the
    physical IDs of the source bands.  The historical acquisition timestamp is
    retained for each residual stream so temporal models can infer its lag.
    """

    if input_mode not in INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode={input_mode!r}; expected one of {INPUT_MODES}"
        )
    if input_mode == "raw":
        return x_dict

    imgs = x_dict["imgs"]
    chn_ids = x_dict["chn_ids"]
    timestamps = x_dict["timestamps"]
    if imgs.ndim != 4 or chn_ids.ndim != 2 or timestamps.ndim != 2:
        raise ValueError(
            "Expected sample tensors imgs=(T,C,H,W), chn_ids=(T,C), "
            f"timestamps=(T,3); got {tuple(imgs.shape)}, {tuple(chn_ids.shape)}, "
            f"{tuple(timestamps.shape)}"
        )
    if len(path_columns) != imgs.shape[0]:
        raise ValueError(
            f"path column count {len(path_columns)} != tensor timepoints {imgs.shape[0]}"
        )

    slot_to_index = {str(column): idx for idx, column in enumerate(path_columns)}
    if "path_t0" in slot_to_index:
        current_index = slot_to_index["path_t0"]
    elif "t0_path" in slot_to_index:
        current_index = slot_to_index["t0_path"]
    else:
        current_index = 0

    missing = [slot for slot in residual_slots if slot not in slot_to_index]
    if missing:
        raise ValueError(
            f"Residual slots {missing} are absent from path columns {tuple(path_columns)}"
        )

    current = imgs[current_index]
    out_imgs: list[torch.Tensor] = []
    out_chn_ids: list[torch.Tensor] = []
    out_timestamps: list[torch.Tensor] = []
    if input_mode in {"current", "current_residual"}:
        out_imgs.append(current)
        out_chn_ids.append(chn_ids[current_index])
        out_timestamps.append(timestamps[current_index])

    if input_mode in {"residual", "current_residual"}:
        for slot in residual_slots:
            history_index = slot_to_index[slot]
            residual = current - imgs[history_index]
            residual = torch.nan_to_num(
                residual, nan=0.0, posinf=0.0, neginf=0.0
            )
            if residual_clip is not None and residual_clip > 0:
                residual = torch.clamp(
                    residual, min=-float(residual_clip), max=float(residual_clip)
                )
            out_imgs.append(residual)
            out_chn_ids.append(chn_ids[current_index])
            out_timestamps.append(timestamps[history_index])

    return {
        "imgs": torch.stack(out_imgs, dim=0),
        "chn_ids": torch.stack(out_chn_ids, dim=0),
        "timestamps": torch.stack(out_timestamps, dim=0),
    }

