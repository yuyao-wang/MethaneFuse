import os
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile as tiff
from torch.utils.data import Dataset

from dinov2.utils.data import extract_wavemus, load_ds_cfg


class _SkipSample(Exception):
    """Internal sentinel exception used to signal that a sample should be skipped."""


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
        path_column: str = "path_t0",
        label_column: str = "label",
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        scale_to_unit: bool = True,
        compute_stats: bool = False,
        compute_stats_subset: Optional[Union[int, float]] = None,
        pad_to_multiple: Optional[int] = 14,
        pad_value: float = 0.0,
        transform=None,
        # ---- robustness options (merged) ----
        max_retries: int = 5,
        skip_invalid_samples: bool = False,
        path_columns_for_validation: Optional[Sequence[str]] = None,
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

            max_retries: If a sample is missing/corrupt, retry by sampling a different index up to this many times.
            skip_invalid_samples: If True, drop rows whose TIFF paths are missing or fail a quick header check.
            path_columns_for_validation: Which columns to validate when skip_invalid_samples=True.
                                       Defaults to (path_column,).
        """
        super().__init__()
        self.transform = transform
        self.id_column = id_column
        self.path_column = path_column
        self.label_column = label_column

        self._skip_invalid_samples = bool(skip_invalid_samples)
        if path_columns_for_validation is None:
            path_columns_for_validation = (self.path_column,)
        self._path_columns_for_validation = tuple(path_columns_for_validation)

        self.normalize_stats = normalize_stats
        self.scale_to_unit = scale_to_unit
        self.compute_stats = compute_stats
        self.compute_stats_subset = compute_stats_subset
        self.pad_to_multiple = pad_to_multiple
        self.pad_value = pad_value
        self._warned_single_channel = False

        self.max_retries = max(1, int(max_retries))

        if isinstance(csv_path, str):
            csv_path = [csv_path]
        dfs = [pd.read_csv(p) for p in csv_path]
        self.df = pd.concat(dfs, ignore_index=True)

        self.chn_ids = extract_wavemus(load_ds_cfg(ds_cfg_name), return_sigmas=full_spectra)
        self.num_classes = 2

        if self._skip_invalid_samples:
            self._filter_invalid_rows()

        if normalize_stats is not None:
            mean, std = normalize_stats
            self._mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)

            std_tensor = torch.tensor(std, dtype=torch.float32)
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
        attempts = 0
        last_exc: Optional[Exception] = None

        while attempts < self.max_retries:
            try:
                row = self.df.iloc[idx]
                label = int(row[self.label_column])
                img_path = row[self.path_column]
                sample_id = row[self.id_column] if self.id_column in row else None

                img = self._load_image(
                    img_path,
                    column_name=self.path_column,
                    sample_id=sample_id,
                )

                x_dict = dict(imgs=img, chn_ids=self.chn_ids)
                if self.transform is not None:
                    x_dict = self.transform(x_dict)

                return x_dict, label

            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts == 1:
                    warnings.warn(
                        f"Skipping corrupt/missing sample at {self.path_column}={img_path}: {exc}. "
                        "Retrying with a different index."
                    )
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for index {idx}") from exc
                idx = np.random.randint(0, len(self.df))

        # Should never reach here
        raise RuntimeError("Unreachable: retry loop exited unexpectedly") from last_exc

    def _load_image(
        self,
        path: str,
        *,
        column_name: Optional[str] = None,
        sample_id: Optional[Union[str, int]] = None,
    ) -> torch.Tensor:
        try:
            img = self._read_image_raw(path)
        except Exception as exc:
            context_bits = []
            if sample_id is not None:
                context_bits.append(f"id={sample_id}")
            if column_name is not None:
                context_bits.append(f"column={column_name}")
            context = f" ({', '.join(context_bits)})" if context_bits else ""
            raise RuntimeError(f"Failed to load image at {path}{context}: {exc}") from exc

        if self._mean is not None and self._std is not None:
            img = (img - self._mean) / self._std

        if self.pad_to_multiple is not None:
            img = self._pad_to_multiple(img, self.pad_to_multiple)

        return img

    def _read_image_raw(self, path: str) -> torch.Tensor:
        if not os.path.isfile(path):
            raise _SkipSample(f"Missing file: {path}")

        try:
            img_np = tiff.imread(path)
        except (tiff.TiffFileError, ValueError, OSError) as exc:
            raise _SkipSample(f"TIFF read failed for {path}: {exc}") from exc

        # torch.from_numpy does not support uint16; cast to float32 up front.
        if img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32)

        exp_c = int(self.chn_ids.shape[0])

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)

        elif img_np.ndim == 3:
            c_first, c_last = img_np.shape[0], img_np.shape[-1]

            if c_first == exp_c:
                pass  # already CHW with expected channels

            elif c_first > exp_c:
                img_np = img_np[:exp_c, ...]  # truncate extra leading channels

            elif c_last == exp_c:
                img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW

            elif c_last > exp_c:
                img_np = np.transpose(img_np, (2, 0, 1))[:exp_c, ...]  # HWC -> CHW then truncate

            elif c_first == 1 and exp_c > 1:
                if not self._warned_single_channel:
                    warnings.warn(
                        f"{path} has 1 channel but {exp_c} expected; repeating the band to match model input."
                    )
                    self._warned_single_channel = True
                img_np = np.repeat(img_np, exp_c, axis=0)

            elif c_last == 1 and exp_c > 1:
                if not self._warned_single_channel:
                    warnings.warn(
                        f"{path} has 1 channel but {exp_c} expected; repeating the band to match model input."
                    )
                    self._warned_single_channel = True
                img_np = np.repeat(img_np, exp_c, axis=2)
                img_np = np.transpose(img_np, (2, 0, 1))

            else:
                raise RuntimeError(
                    f"Unexpected image shape {img_np.shape} for {path}; expected channel dim {exp_c} "
                    "either first (C,H,W) or last (H,W,C)."
                )

        else:
            raise RuntimeError(f"Unsupported image dimensions {img_np.shape} for {path}")

        img = torch.from_numpy(img_np).to(dtype=torch.float32)

        if self.scale_to_unit:
            img = img / 65535.0

        return img

    def _filter_invalid_rows(self) -> None:
        valid_indices: List[int] = []
        bad_entries: List[Tuple[Union[str, int], str, str, str]] = []

        for idx, row in self.df.iterrows():
            ok = True
            sample_id = row[self.id_column] if self.id_column in row else idx

            for column in self._path_columns_for_validation:
                path = row[column]
                is_valid, reason = self._quick_validate_tiff(path)
                if not is_valid:
                    bad_entries.append((sample_id, column, path, reason))
                    ok = False
                    break

            if ok:
                valid_indices.append(idx)

        if bad_entries:
            preview = "\n".join(
                f"  id={sid}, column={col}, path={path}, reason={reason}"
                for sid, col, path, reason in bad_entries[:5]
            )
            warnings.warn(
                f"Skipping {len(bad_entries)} samples with unreadable/missing TIFF files. Examples:\n{preview}"
            )

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    @staticmethod
    def _quick_validate_tiff(path: str) -> Tuple[bool, str]:
        try:
            with open(path, "rb") as fh:
                header = fh.read(4)
        except FileNotFoundError:
            return False, "missing"
        except OSError as exc:
            return False, f"os_error:{exc}"

        if len(header) < 4:
            return False, "empty_header"
        if header[:2] not in (b"II", b"MM", b"EP"):
            return False, f"bad_magic:{header[:2]!r}"
        return True, ""

    def _compute_dataset_stats(
        self, subset: Optional[Union[int, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-channel mean/std after optional scaling (no padding)."""
        idxs = list(range(len(self.df)))
        if subset is not None:
            if isinstance(subset, float):
                subset = int(len(idxs) * subset)
            subset = max(1, min(len(idxs), subset))
            idxs = idxs[:subset]

        c = int(self.chn_ids.shape[0])
        sum_c = torch.zeros(c, dtype=torch.double)
        sumsq_c = torch.zeros(c, dtype=torch.double)
        count = 0

        for i in idxs:
            path = self.df.iloc[i][self.path_column]
            img = self._read_image_raw(path).double()
            count += int(img.shape[1] * img.shape[2])
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


class S2TemporalCsvDataset(S2CsvDataset):
    """
    Temporal extension that returns a list of three timepoint inputs while sharing the same
    Panopticon preprocessing (normalization, padding, channel embeddings).

    Expected CSV columns by default:
        - image_path: t0 (leak day)
        - s2_pre_path: t-90
        - s2_pre_pre_path: t-360
    """

    def __init__(
        self,
        csv_path: Union[str, Iterable[str]],
        *,
        ds_cfg_name: str = "s2_12band",
        full_spectra: bool = False,
        id_column: str = "id",
        path_columns: Sequence[str] = ("image_path", "s2_pre_path", "s2_pre_pre_path"),
        label_column: str = "label",
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        scale_to_unit: bool = True,
        compute_stats: bool = False,
        compute_stats_subset: Optional[Union[int, float]] = None,
        pad_to_multiple: Optional[int] = 14,
        pad_value: float = 0.0,
        transform=None,
        # ---- robustness options (merged) ----
        max_retries: int = 5,
        skip_invalid_samples: bool = False,
    ):
        """
        Args:
            path_columns: Ordered sequence of CSV columns for each timepoint.
            skip_invalid_samples: If True, drop rows where any referenced TIFF path is missing or invalid.
            max_retries: If a temporal sample is corrupt/missing, retry by sampling a different index.
        """
        if len(path_columns) != 3:
            raise ValueError(f"Expected exactly 3 time columns, got {len(path_columns)}: {path_columns}")

        super().__init__(
            csv_path=csv_path,
            ds_cfg_name=ds_cfg_name,
            full_spectra=full_spectra,
            id_column=id_column,
            path_column=path_columns[0],
            label_column=label_column,
            normalize_stats=normalize_stats,
            scale_to_unit=scale_to_unit,
            compute_stats=compute_stats,
            compute_stats_subset=compute_stats_subset,
            pad_to_multiple=pad_to_multiple,
            pad_value=pad_value,
            transform=None,  # apply transform manually per timepoint
            max_retries=max_retries,
            skip_invalid_samples=skip_invalid_samples,
            path_columns_for_validation=path_columns,
        )

        self.path_columns = tuple(path_columns)
        self.transform_each = transform

    def __getitem__(self, idx):
        attempts = 0
        last_exc: Optional[Exception] = None

        while attempts < self.max_retries:
            try:
                row = self.df.iloc[idx]
                label = int(row[self.label_column])
                sample_id = row[self.id_column] if self.id_column in row else None

                x_list = []
                for col in self.path_columns:
                    path = row[col]
                    img = self._load_image(path, column_name=col, sample_id=sample_id)

                    x_dict = dict(imgs=img, chn_ids=self.chn_ids)
                    if self.transform_each is not None:
                        x_dict = self.transform_each(x_dict)
                    x_list.append(x_dict)

                return x_list, label

            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts == 1:
                    warnings.warn(
                        f"Skipping corrupt temporal sample at row {idx}: {exc}. Retrying with a different index."
                    )
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
                idx = np.random.randint(0, len(self.df))

        raise RuntimeError("Unreachable: retry loop exited unexpectedly") from last_exc
