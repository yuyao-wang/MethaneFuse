import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dinov2.utils.data import extract_wavemus, load_ds_cfg


class S5pNpzDataset(Dataset):
    """
    CSV-based Sentinel-5P dataset for Panopticon finetuning.

    Expected CSV columns (customizable): label, image_path, id.
    Images are stored as .npz files containing a HxWxC or CxHxW array.
    """

    def __init__(
        self,
        csv_path: Union[str, Iterable[str]],
        *,
        ds_cfg_name: Optional[str] = None,
        chn_ids: Optional[Sequence[float]] = None,
        full_spectra: bool = False,
        id_column: str = "id",
        path_column: str = "image_path",
        label_column: str = "label",
        data_key: Optional[str] = None,
        chn_ids_key: Optional[str] = "chn_ids",
        channels_first: bool = True,
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        scale_to_unit: bool = False,
        scale_value: float = 65535.0,
        compute_stats: bool = False,
        compute_stats_subset: Optional[Union[int, float]] = None,
        pad_to_multiple: Optional[int] = 14,
        pad_value: float = 0.0,
        transform=None,
        default_chn_id_value: float = 0.0,
        allow_pickle: bool = False,
        nan_to_num: Optional[float] = None,
    ):
        """
        Args:
            csv_path: Path or iterable of paths to CSV files to concatenate.
            ds_cfg_name: Dataset config name under dinov2/configs/data for chn_ids. If None, chn_ids are inferred.
            chn_ids: Optional explicit channel IDs (length C or Cx2). Overrides ds_cfg_name.
            full_spectra: If True and ds_cfg_name is provided, include spectral stds in chn_ids.
            id_column: Column name for sample id (kept for reference).
            path_column: Column name with absolute/relative paths to NPZ files.
            label_column: Column name with integer class labels.
            data_key: Optional key inside the NPZ; if None, the first array entry is used.
            chn_ids_key: Optional key inside the NPZ storing per-sample chn_ids (overrides ds_cfg_name if present).
            channels_first: If True, expect CxHxW inside the NPZ; if False, transpose HWC to CHW.
            normalize_stats: (mean, std) to apply per channel. If None, optional scaling is used.
            scale_to_unit: If True and normalize_stats is None, divide by scale_value.
            scale_value: Divisor used when scale_to_unit is True.
            pad_to_multiple: If set, pad H/W to the next multiple of this value to avoid patch truncation.
            pad_value: Value used when padding.
            transform: Optional augmentation callable operating on x_dict.
            compute_stats: If True and normalize_stats is None, compute mean/std over the dataset (after optional scaling).
            compute_stats_subset: Limit samples for stats (int count or float fraction).
            default_chn_id_value: Constant channel id used when none are provided (avoids SRF dependency for single-channel grids).
            allow_pickle: If True, allow loading NPZ files that contain pickled objects (set only if you trust the data source).
            nan_to_num: If set, replace NaN/Inf values in NPZ arrays with this constant.
        """
        super().__init__()
        self.transform = transform
        self.id_column = id_column
        self.path_column = path_column
        self.label_column = label_column
        self.normalize_stats = normalize_stats
        self.scale_to_unit = scale_to_unit
        self.scale_value = scale_value
        self.compute_stats = compute_stats
        self.compute_stats_subset = compute_stats_subset
        self.pad_to_multiple = pad_to_multiple
        self.pad_value = pad_value
        self.data_key = data_key
        self.chn_ids_key = chn_ids_key
        self.channels_first = channels_first
        self.default_chn_id_value = default_chn_id_value
        self.allow_pickle = allow_pickle
        self.nan_to_num = nan_to_num
        self._stats_source: Optional[str] = None

        if isinstance(csv_path, str):
            csv_path = [csv_path]
        dfs = [pd.read_csv(p) for p in csv_path]
        self.df = pd.concat(dfs, ignore_index=True)

        if chn_ids is not None:
            self.chn_ids = torch.as_tensor(chn_ids)
        elif ds_cfg_name is not None:
            self.chn_ids = extract_wavemus(load_ds_cfg(ds_cfg_name), return_sigmas=full_spectra)
        else:
            self.chn_ids = None

        self.num_classes = 2

        if normalize_stats is not None:
            mean, std = normalize_stats
            self._mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            std_tensor = torch.tensor(std, dtype=torch.float32)
            if torch.any(std_tensor <= 0):
                warnings.warn("Found non-positive std values; clamping to 1e-6 to avoid NaNs.")
            std_tensor = torch.clamp(std_tensor, min=1e-6)
            self._std = std_tensor.view(-1, 1, 1)
            self._stats_source = "provided"
        else:
            self._mean = None
            self._std = None
            if self.compute_stats:
                mean, std = self._compute_dataset_stats(subset=self.compute_stats_subset)
                self._mean = mean.view(-1, 1, 1)
                std = torch.clamp(std, min=1e-6)
                self._std = std.view(-1, 1, 1)
                self._stats_source = "computed"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row[self.label_column])
        img_path = row[self.path_column]

        img, chn_ids = self._load_image(img_path)
        x_dict = dict(imgs=img, chn_ids=chn_ids)
        if self.transform is not None:
            x_dict = self.transform(x_dict)

        return x_dict, label

    def _load_image(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        img, file_chn_ids = self._read_image_raw(path)
        if self._mean is not None and self._std is not None:
            img = (img - self._mean) / self._std
        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)
        chn_ids = self._resolve_chn_ids(file_chn_ids, img.shape[0])
        return img, chn_ids

    def _resolve_chn_ids(self, file_chn_ids: Optional[torch.Tensor], num_channels: int) -> torch.Tensor:
        if self.chn_ids is not None:
            if self.chn_ids.shape[0] == num_channels:
                return self.chn_ids
            if self.chn_ids.shape[0] == 1 and num_channels > 1:
                return self.chn_ids.repeat(num_channels)
            raise ValueError(f"Configured chn_ids length {self.chn_ids.shape[0]} does not match channels {num_channels}")
        if file_chn_ids is not None:
            if file_chn_ids.shape[0] == num_channels:
                return file_chn_ids
            if file_chn_ids.shape[0] == 1 and num_channels > 1:
                return file_chn_ids.repeat(num_channels)
            raise ValueError(f"chn_ids in NPZ length {file_chn_ids.shape[0]} does not match channels {num_channels}")
        return torch.full((num_channels,), fill_value=self.default_chn_id_value, dtype=torch.int16)

    def _read_image_raw(self, path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with np.load(path, allow_pickle=self.allow_pickle) as npz:
            if self.data_key is not None:
                if self.data_key not in npz:
                    raise KeyError(f"Key '{self.data_key}' not found in NPZ file {path}")
                img_np = np.array(npz[self.data_key])
            else:
                if len(npz.files) == 0:
                    raise ValueError(f"No arrays found in NPZ file {path}")
                img_np = np.array(npz[npz.files[0]])

            file_chn_ids = None
            if self.chn_ids_key is not None and self.chn_ids_key in npz:
                file_chn_ids = torch.as_tensor(npz[self.chn_ids_key])

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3:
            if not self.channels_first:
                img_np = np.transpose(img_np, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported NPZ array shape {img_np.shape} in {path}")

        img = torch.from_numpy(img_np).to(dtype=torch.float32)

        if self.nan_to_num is not None:
            img = torch.nan_to_num(img, nan=self.nan_to_num, posinf=self.nan_to_num, neginf=self.nan_to_num)

        if self.scale_to_unit:
            img = img / float(self.scale_value)

        return img, file_chn_ids

    def _compute_dataset_stats(self, subset: Optional[Union[int, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-channel mean/std after optional scaling (no padding)."""
        idxs = list(range(len(self.df)))
        if subset is not None:
            if isinstance(subset, float):
                subset = int(len(idxs) * subset)
            subset = max(1, min(len(idxs), subset))
            idxs = idxs[:subset]

        # Infer channel count from the first sample.
        first_img, _ = self._read_image_raw(self.df.iloc[idxs[0]][self.path_column])
        c = first_img.shape[0]
        sum_c = torch.zeros(c, dtype=torch.double)
        sumsq_c = torch.zeros(c, dtype=torch.double)
        count = 0

        for i in idxs:
            path = self.df.iloc[i][self.path_column]
            img, _ = self._read_image_raw(path)
            img = img.double()
            count += img.shape[1] * img.shape[2]
            sum_c += img.sum(dim=(1, 2))
            sumsq_c += (img * img).sum(dim=(1, 2))

        mean = (sum_c / count).float()
        std = torch.sqrt((sumsq_c / count) - (mean.double() ** 2)).float()
        return mean, std

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

    def get_normalize_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (mean, std) tensors if available."""
        if self._mean is None or self._std is None:
            return None
        return self._mean.squeeze().cpu(), self._std.squeeze().cpu()

    def get_stats_source(self) -> Optional[str]:
        """Return 'provided', 'computed', or None for normalization stats origin."""
        return self._stats_source


class S5pTemporalNpzDataset(S5pNpzDataset):
    """
    Temporal extension that returns a list of three timepoint inputs while sharing the same
    Panopticon preprocessing (normalization, padding, channel embeddings).

    Expected CSV columns by default:
        - image_path: t0
        - s2_pre_path: t-90
        - s2_pre_pre_path: t-360
    """

    def __init__(
        self,
        csv_path: Union[str, Iterable[str]],
        *,
        ds_cfg_name: Optional[str] = None,
        chn_ids: Optional[Sequence[float]] = None,
        full_spectra: bool = False,
        id_column: str = "id",
        path_columns: Sequence[str] = ("image_path", "s5p_pre_path", "s5p_pre_pre_path"),
        label_column: str = "label",
        data_key: Optional[str] = None,
        chn_ids_key: Optional[str] = "chn_ids",
        channels_first: bool = True,
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        scale_to_unit: bool = False,
        scale_value: float = 65535.0,
        compute_stats: bool = False,
        compute_stats_subset: Optional[Union[int, float]] = None,
        pad_to_multiple: Optional[int] = 14,
        pad_value: float = 0.0,
        transform=None,
        default_chn_id_value: float = 0.0,
        allow_pickle: bool = False,
        stacked_time: bool = False,
        nan_to_num: Optional[float] = None,
    ):
        """
        Args:
            path_columns: Ordered sequence of CSV columns for each timepoint.
            stacked_time: If True, load a single NPZ containing stacked time slices on the channel dimension (e.g., [T,H,W]).
            allow_pickle: If True, allow loading NPZ files that contain pickled objects (set only if you trust the data source).
            nan_to_num: If set, replace NaN/Inf values in NPZ arrays with this constant.
        """
        if stacked_time:
            if len(path_columns) != 1:
                raise ValueError(f"stacked_time=True expects a single path column, got {len(path_columns)}: {path_columns}")
        else:
            if len(path_columns) != 3:
                raise ValueError(f"Expected exactly 3 time columns, got {len(path_columns)}: {path_columns}")

        super().__init__(
            csv_path=csv_path,
            ds_cfg_name=ds_cfg_name,
            chn_ids=chn_ids,
            full_spectra=full_spectra,
            id_column=id_column,
            path_column=path_columns[0],
            label_column=label_column,
            data_key=data_key,
            chn_ids_key=chn_ids_key,
            channels_first=channels_first,
            normalize_stats=normalize_stats,
            scale_to_unit=scale_to_unit,
            scale_value=scale_value,
            compute_stats=compute_stats,
            compute_stats_subset=compute_stats_subset,
            pad_to_multiple=pad_to_multiple,
            pad_value=pad_value,
            transform=None,  # apply transform manually per timepoint
            default_chn_id_value=default_chn_id_value,
            allow_pickle=allow_pickle,
            nan_to_num=nan_to_num,
        )
        self.path_columns = tuple(path_columns)
        self.transform_each = transform
        self.stacked_time = stacked_time

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row[self.label_column])

        if self.stacked_time:
            path = row[self.path_columns[0]]
            img, chn_ids = self._load_image(path)
            if img.ndim != 3:
                raise ValueError(f"Expected stacked time tensor with shape (T,H,W) or (C,H,W); got {img.shape}")
            x_list = []
            for t in range(img.shape[0]):
                slice_img = img[t : t + 1]  # keep channel dim
                slice_chn = chn_ids[t : t + 1] if chn_ids.ndim > 1 else chn_ids[t : t + 1]
                x_dict = dict(imgs=slice_img, chn_ids=slice_chn)
                if self.transform_each is not None:
                    x_dict = self.transform_each(x_dict)
                x_list.append(x_dict)
        else:
            x_list = []
            for col in self.path_columns:
                path = row[col]
                img, chn_ids = self._load_image(path)
                x_dict = dict(imgs=img, chn_ids=chn_ids)
                if self.transform_each is not None:
                    x_dict = self.transform_each(x_dict)
                x_list.append(x_dict)

        return x_list, label
