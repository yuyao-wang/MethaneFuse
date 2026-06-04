from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import tifffile as tiff

from thirdparty.dinov2.data.datasets.s2_csv import S2CsvDataset


class Landsat89CsvDataset(S2CsvDataset):
    """
    CSV-based dataset for Landsat 8/9 surface reflectance stacks.

    - Expected input bands: SR_B1..SR_B7 (QA bands are ignored if present).
    - Column names mirror S2CsvDataset: label, image_path, id (customizable via args).
    """

    def __init__(
        self,
        csv_path: Union[str, Iterable[str]],
        *,
        ds_cfg_name: str = "landsat89_7band",
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
        expected_bands: Optional[int] = None,
        drop_extra_bands: bool = True,
    ):
        self.expected_bands = expected_bands
        self.drop_extra_bands = drop_extra_bands

        # Defer stats computation until after expected_bands is resolved to avoid None comparisons.
        compute_stats_flag = compute_stats and normalize_stats is None
        super().__init__(
            csv_path=csv_path,
            ds_cfg_name=ds_cfg_name,
            full_spectra=False,
            id_column=id_column,
            path_column=path_column,
            label_column=label_column,
            normalize_stats=normalize_stats,
            scale_to_unit=scale_to_unit,
            compute_stats=False,  # handled manually below
            compute_stats_subset=None,
            pad_to_multiple=pad_to_multiple,
            pad_value=pad_value,
            transform=transform,
        )
        # Default to the length of chn_ids (from ds config) when not provided.
        if self.expected_bands is None:
            self.expected_bands = int(self.chn_ids.shape[0])
        if compute_stats_flag:
            mean, std = self._compute_dataset_stats(subset=compute_stats_subset)
            mean = mean.view(-1, 1, 1)
            std = torch.clamp(std, min=1e-6).view(-1, 1, 1)
            self._mean = mean
            self._std = std

    def _read_image_raw(self, path: str) -> torch.Tensor:
        img_np = tiff.imread(path)

        if img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32)

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, 0)
        elif img_np.ndim == 3:
            # Move the most plausible channel axis to the front.
            if self.expected_bands in img_np.shape:
                channel_axis = list(img_np.shape).index(self.expected_bands)
            else:
                channel_axis = int(np.argmin(img_np.shape))
            img_np = np.moveaxis(img_np, channel_axis, 0)
        else:
            raise ValueError(f"Unexpected array shape {img_np.shape} from {path}")

        if img_np.shape[0] > self.expected_bands:
            if not self.drop_extra_bands:
                raise ValueError(
                    f"Found {img_np.shape[0]} channels, expected {self.expected_bands} for {path}; "
                    f"set drop_extra_bands=True to discard QA bands."
                )
            img_np = img_np[: self.expected_bands, ...]
        elif img_np.shape[0] < self.expected_bands:
            raise ValueError(
                f"Image at {path} has {img_np.shape[0]} channels; at least {self.expected_bands} are required."
            )

        img = torch.from_numpy(img_np).to(dtype=torch.float32)

        if self.scale_to_unit:
            img = img / 65535.0

        return img
