"""Segmentation dataset wrappers for MethaneFuse sensor-specific masks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.multisensor import TriSensorTemporalCsvDataset

@dataclass(frozen=True)
class TaskConfig:
    name: str
    sensor_key: str
    image_columns: Tuple[str, str, str]
    mask_column: str


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "s2": TaskConfig(
        name="s2",
        sensor_key="s2",
        image_columns=("s2_0_path", "s2_90_path", "s2_360_path"),
        mask_column="s2_plume_path",
    ),
    "l89": TaskConfig(
        name="l89",
        sensor_key="l89",
        image_columns=("l89_0_path", "l89_90_path", "l89_360_path"),
        mask_column="l89_plume_path",
    ),
    "emit": TaskConfig(
        name="emit",
        sensor_key="wv3",
        image_columns=("emit_0_path", "emit_90_path", "emit_360_path"),
        mask_column="emit_plume_path",
    ),
}


def _is_missing(value: object) -> bool:
    text = str(value).strip().lower()
    return text in ("", "nan", "none", "null")


def parse_tasks(tasks_text: str) -> List[TaskConfig]:
    names = [x.strip().lower() for x in tasks_text.split(",") if x.strip()]
    if not names:
        raise ValueError("--tasks cannot be empty")
    out: List[TaskConfig] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        if name not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task '{name}'. Available: {sorted(TASK_CONFIGS)}")
        out.append(TASK_CONFIGS[name])
        seen.add(name)
    return out


class SingleSensorSegmentationDataset(Dataset):
    """Wrap a wide-table multi-sensor dataset into one sensor-specific seg task."""

    def __init__(self, base_ds: TriSensorTemporalCsvDataset, task: TaskConfig):
        self.base_ds = base_ds
        self.task = task
        self.max_retries = max(1, int(getattr(base_ds, "max_retries", 3)))
        self.indices = self._collect_valid_indices()
        if len(self.indices) == 0:
            raise ValueError(f"No valid rows found for task={task.name}")

    def _collect_valid_indices(self) -> List[int]:
        idxs: List[int] = []
        cols = list(self.task.image_columns) + [self.task.mask_column]
        for i, row in self.base_ds.df.iterrows():
            ok = True
            for col in cols:
                if col not in self.base_ds.df.columns or _is_missing(row.get(col, "")):
                    ok = False
                    break
            if ok:
                idxs.append(int(i))
        return idxs

    def __len__(self) -> int:
        return len(self.indices)

    def _load_mask(self, row, imgs_hw: Tuple[int, int]) -> torch.Tensor:
        mask_path = str(row[self.task.mask_column]).strip()
        mask_path = self.base_ds._maybe_cache_path(mask_path)
        arr = tiff.imread(mask_path)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Mask must be 2D or single-band 3D, got shape={arr.shape} at {mask_path}")
        mask = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (mask > 0).float()
        if self.base_ds.pad_to_multiple is not None:
            mask = self.base_ds._pad_to_multiple(mask.unsqueeze(0), self.base_ds.pad_to_multiple).squeeze(0)
        if tuple(mask.shape) != tuple(imgs_hw):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=imgs_hw,
                mode="nearest",
            ).squeeze(0).squeeze(0)
        return mask

    def __getitem__(self, idx: int):
        attempts = 0
        current_idx = idx
        last_exc: Optional[Exception] = None
        while attempts < self.max_retries:
            row_idx = self.indices[current_idx]
            row = self.base_ds.df.iloc[row_idx]
            try:
                x_list = [
                    self.base_ds._load_temporal_frame(
                        row=row,
                        column_name=col,
                        sensor=self.task.sensor_key,
                        sample_id=row_idx,
                    )
                    for col in self.task.image_columns
                ]
                imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
                chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
                mask = self._load_mask(row=row, imgs_hw=(imgs.shape[1], imgs.shape[2]))
                sample_id = row.get("id", row_idx)
                return {"imgs": imgs, "chn_ids": chn_ids}, mask.unsqueeze(0), sample_id
            except Exception as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    raise RuntimeError(
                        f"Failed to load sample after {self.max_retries} retries for task={self.task.name}, "
                        f"row_idx={row_idx}"
                    ) from exc
                current_idx = random.randint(0, len(self.indices) - 1)
        raise RuntimeError("Unreachable") from last_exc


def segmentation_collate_fn(batch):
    x_dicts: List[Dict[str, torch.Tensor]] = []
    masks: List[torch.Tensor] = []
    sample_ids: List[object] = []
    for x_dict, mask, sample_id in batch:
        x_dicts.append(x_dict)
        masks.append(mask)
        sample_ids.append(sample_id)

    max_channels = max(x["imgs"].shape[0] for x in x_dicts)
    max_h = max(x["imgs"].shape[1] for x in x_dicts)
    max_w = max(x["imgs"].shape[2] for x in x_dicts)

    padded_imgs = []
    padded_chn_ids = []
    padded_masks = []
    for x_dict, mask in zip(x_dicts, masks):
        img = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        img = F.pad(img, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))
        pad_c = max_channels - img.shape[0]
        if pad_c > 0:
            img = torch.cat([img, torch.zeros((pad_c, max_h, max_w), dtype=img.dtype)], dim=0)
            chn_ids = torch.cat(
                [chn_ids, torch.zeros((pad_c, *chn_ids.shape[1:]), dtype=chn_ids.dtype)],
                dim=0,
            )
        padded_imgs.append(img)
        padded_chn_ids.append(chn_ids)
        padded_masks.append(mask)

    batched_x = {"imgs": torch.stack(padded_imgs), "chn_ids": torch.stack(padded_chn_ids)}
    batched_mask = torch.stack(padded_masks)
    return batched_x, batched_mask, sample_ids
