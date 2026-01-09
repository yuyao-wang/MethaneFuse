import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile as tiff
import warnings
from torch.utils.data import Dataset

from dinov2.utils.data import extract_wavemus, load_ds_cfg


class S2CsvDataset(Dataset):
    """
    Minimal CSV-based Sentinel-2 dataset for Panopticon finetuning.

    Expected CSV columns (customizable): label, image_path, id.
    """

    def __init__(
        self,
        csv_path: Union[str, Iterable[str]],
        *,
        ds_cfg_name: str = "s2_12band",
        full_spectra: bool = False,
        id_column: str = "id",
        path_column: str = "image_path",
        label_column: str = "label",
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        scale_to_unit: bool = True,
        compute_stats: bool = False,
        compute_stats_subset: Optional[Union[int, float]] = None,
        pad_to_multiple: Optional[int] = 14,
        pad_value: float = 0.0,
        transform=None,
    ):
        """
        Args:
            csv_path: Path or iterable of paths to CSV files to concatenate.
            ds_cfg_name: Dataset config name under dinov2/configs/data for chn_ids.
            full_spectra: If True, include spectral stds in chn_ids.
            id_column: Column name for sample id (kept for reference).
            path_column: Column name with absolute/relative paths to TIFFs.
            label_column: Column name with integer class labels.
            normalize_stats: (mean, std) to apply per channel. If None, optional scaling is used.
            scale_to_unit: If True and normalize_stats is None, divide by 65535.0 for uint16 inputs.
            pad_to_multiple: If set, pad H/W to the next multiple of this value to avoid patch truncation.
            pad_value: Value used when padding.
            transform: Optional augmentation callable operating on x_dict.
            compute_stats: If True and normalize_stats is None, compute mean/std over the dataset (after optional scaling).
            compute_stats_subset: Limit samples for stats (int count or float fraction).
        """
        super().__init__()
        self.transform = transform
        self.id_column = id_column
        self.path_column = path_column
        self.label_column = label_column
        self.normalize_stats = normalize_stats
        self.scale_to_unit = scale_to_unit
        self.compute_stats = compute_stats
        self.compute_stats_subset = compute_stats_subset
        self.pad_to_multiple = pad_to_multiple
        self.pad_value = pad_value

        if isinstance(csv_path, str):
            csv_path = [csv_path]
        dfs = [pd.read_csv(p) for p in csv_path]
        self.df = pd.concat(dfs, ignore_index=True)

        self.chn_ids = extract_wavemus(load_ds_cfg(ds_cfg_name), return_sigmas=full_spectra)
        self.num_classes = 2

        if normalize_stats is not None:
            mean, std = normalize_stats
            self._mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            std_tensor = torch.tensor(std, dtype=torch.float32)
            # Avoid division by zero when std contains zeros.
            if torch.any(std_tensor <= 0):
                warnings.warn("Found non-positive std values; clamping to 1e-6 to avoid NaNs.")
            std_tensor = torch.clamp(std_tensor, min=1e-6)
            self._std = std_tensor.view(-1, 1, 1)
        else:
            self._mean = None
            self._std = None
            if self.compute_stats:
                mean, std = self._compute_dataset_stats(subset=self.compute_stats_subset)
                self._mean = mean.view(-1, 1, 1)
                std = torch.clamp(std, min=1e-6)
                self._std = std.view(-1, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row[self.label_column])
        img_path = row[self.path_column]
        img = self._load_image(img_path)

        x_dict = dict(imgs=img, chn_ids=self.chn_ids)
        if self.transform is not None:
            x_dict = self.transform(x_dict)

        return x_dict, label

    def _load_image(self, path: str) -> torch.Tensor:
        img = self._read_image_raw(path)
        if self._mean is not None and self._std is not None:
            img = (img - self._mean) / self._std
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)
        return img

    def _read_image_raw(self, path: str) -> torch.Tensor:
        img_np = tiff.imread(path)

        # torch.from_numpy does not support uint16; cast to float32 up front.
        if img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32)

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3 and img_np.shape[0] != self.chn_ids.shape[0]:
            img_np = np.transpose(img_np, (2, 0, 1))

        img = torch.from_numpy(img_np).to(dtype=torch.float32)

        if self.scale_to_unit:
            img = img / 65535.0

        return img

    def _pad_to_multiple(self, img: torch.Tensor, multiple: int) -> torch.Tensor:
        _, h, w = img.shape
        target_h = int(np.ceil(h / multiple) * multiple)
        target_w = int(np.ceil(w / multiple) * multiple)
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return img

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        return F.pad(img, padding, value=self.pad_value)

    def _compute_dataset_stats(self, subset: Optional[Union[int, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-channel mean/std after optional scaling (no padding)."""
        idxs = list(range(len(self.df)))
        if subset is not None:
            if isinstance(subset, float):
                subset = int(len(idxs) * subset)
            subset = max(1, min(len(idxs), subset))
            idxs = idxs[:subset]

        c = self.chn_ids.shape[0]
        sum_c = torch.zeros(c, dtype=torch.double)
        sumsq_c = torch.zeros(c, dtype=torch.double)
        count = 0

        for i in idxs:
            path = self.df.iloc[i][self.path_column]
            img = self._read_image_raw(path).double()
            count += img.shape[1] * img.shape[2]
            sum_c += img.sum(dim=(1, 2))
            sumsq_c += (img * img).sum(dim=(1, 2))

        mean = (sum_c / count).float()
        std = torch.sqrt((sumsq_c / count) - (mean.double() ** 2)).float()
        return mean, std

    def get_normalize_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (mean, std) tensors if available."""
        if self._mean is None or self._std is None:
            return None
        return self._mean.squeeze().cpu(), self._std.squeeze().cpu()
