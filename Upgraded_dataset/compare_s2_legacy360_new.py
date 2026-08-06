import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F


TIMEPOINTS = ("0", "90", "360")


def read_image(path: str) -> np.ndarray:
    image = np.asarray(tifffile.imread(path), dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D TIFF at {path}, got {image.shape}")
    if image.shape[0] != 12 and image.shape[-1] == 12:
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] != 12:
        raise ValueError(f"Expected 12 channels at {path}, got {image.shape}")
    return image


def resize_image(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).numpy()


def apply_legacy_t0_contract(image: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(image)
    retained_channels = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    retained = image[retained_channels]
    transformed[retained_channels] = np.where(retained == 0, 0, retained + 1000)
    transformed[7] = np.where(image[8] == 0, 0, image[8] + 1000)
    return transformed


def compare_row(item: tuple[int, dict, dict]) -> tuple[int, dict]:
    row_id, new_row, old_row = item
    result = {}
    for timepoint in TIMEPOINTS:
        new_image = read_image(new_row[f"s2_{timepoint}_path"])
        old_image = read_image(old_row[f"s2_{timepoint}_path"])
        legacy_compat_image = (
            apply_legacy_t0_contract(new_image) if timepoint == "0" else new_image
        )
        if new_image.shape != old_image.shape:
            new_image = resize_image(new_image, old_image.shape[-2:])
            legacy_compat_image = resize_image(
                legacy_compat_image, old_image.shape[-2:]
            )
        difference = new_image - old_image
        legacy_compat_difference = legacy_compat_image - old_image
        result[timepoint] = {
            "new_sum": new_image.sum(axis=(1, 2)),
            "new_sumsq": np.square(new_image).sum(axis=(1, 2)),
            "old_sum": old_image.sum(axis=(1, 2)),
            "old_sumsq": np.square(old_image).sum(axis=(1, 2)),
            "pixel_count": new_image.shape[1] * new_image.shape[2],
            "abs_sum": np.abs(difference).sum(),
            "equal_pixels": int(np.count_nonzero(difference == 0)),
            "legacy_compat_abs_sum": np.abs(legacy_compat_difference).sum(),
            "legacy_compat_equal_pixels": int(
                np.count_nonzero(legacy_compat_difference == 0)
            ),
            "element_count": difference.size,
            "new_zero_pixels": int(np.count_nonzero(new_image == 0)),
            "old_zero_pixels": int(np.count_nonzero(old_image == 0)),
            "new_min": float(new_image.min()),
            "new_max": float(new_image.max()),
            "old_min": float(old_image.min()),
            "old_max": float(old_image.max()),
        }
    return row_id, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-csv", required=True)
    parser.add_argument("--old-train-csv", required=True)
    parser.add_argument("--old-test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    new_frame = pd.read_csv(args.new_csv).set_index("id", drop=False)
    old_frame = pd.concat(
        [pd.read_csv(args.old_train_csv), pd.read_csv(args.old_test_csv)],
        ignore_index=True,
    ).drop_duplicates("id").set_index("id", drop=False)
    common_ids = new_frame.index.intersection(old_frame.index)
    sample_count = min(max(1, args.samples), len(common_ids))
    sampled_ids = (
        pd.Series(common_ids)
        .sample(n=sample_count, random_state=args.seed, replace=False)
        .astype(int)
        .tolist()
    )
    items = [
        (
            row_id,
            new_frame.loc[row_id].to_dict(),
            old_frame.loc[row_id].to_dict(),
        )
        for row_id in sampled_ids
    ]

    aggregates = {
        timepoint: {
            "new_sum": np.zeros(12, dtype=np.float64),
            "new_sumsq": np.zeros(12, dtype=np.float64),
            "old_sum": np.zeros(12, dtype=np.float64),
            "old_sumsq": np.zeros(12, dtype=np.float64),
            "pixel_count": 0,
            "abs_sum": 0.0,
            "equal_pixels": 0,
            "legacy_compat_abs_sum": 0.0,
            "legacy_compat_equal_pixels": 0,
            "element_count": 0,
            "new_zero_pixels": 0,
            "old_zero_pixels": 0,
            "new_min": float("inf"),
            "new_max": float("-inf"),
            "old_min": float("inf"),
            "old_max": float("-inf"),
        }
        for timepoint in TIMEPOINTS
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (_, result) in enumerate(executor.map(compare_row, items), start=1):
            for timepoint in TIMEPOINTS:
                aggregate = aggregates[timepoint]
                current = result[timepoint]
                for key in ("new_sum", "new_sumsq", "old_sum", "old_sumsq"):
                    aggregate[key] += current[key]
                for key in (
                    "pixel_count",
                    "abs_sum",
                    "equal_pixels",
                    "legacy_compat_abs_sum",
                    "legacy_compat_equal_pixels",
                    "element_count",
                    "new_zero_pixels",
                    "old_zero_pixels",
                ):
                    aggregate[key] += current[key]
                for key in ("new_min", "old_min"):
                    aggregate[key] = min(aggregate[key], current[key])
                for key in ("new_max", "old_max"):
                    aggregate[key] = max(aggregate[key], current[key])
            if index % 100 == 0 or index == len(items):
                print(f"[Compare] processed {index}/{len(items)} rows", flush=True)

    output = {
        "sampled_rows": sample_count,
        "common_rows": len(common_ids),
        "seed": args.seed,
        "timepoints": {},
    }
    for timepoint, aggregate in aggregates.items():
        pixel_count = aggregate["pixel_count"]
        new_mean = aggregate["new_sum"] / pixel_count
        old_mean = aggregate["old_sum"] / pixel_count
        new_std = np.sqrt(
            np.maximum(aggregate["new_sumsq"] / pixel_count - np.square(new_mean), 0.0)
        )
        old_std = np.sqrt(
            np.maximum(aggregate["old_sumsq"] / pixel_count - np.square(old_mean), 0.0)
        )
        output["timepoints"][timepoint] = {
            "new_mean": new_mean.tolist(),
            "new_std": new_std.tolist(),
            "old_mean": old_mean.tolist(),
            "old_std": old_std.tolist(),
            "mean_absolute_pixel_difference": aggregate["abs_sum"]
            / aggregate["element_count"],
            "exact_pixel_fraction": aggregate["equal_pixels"]
            / aggregate["element_count"],
            "legacy_compat_mean_absolute_pixel_difference": aggregate[
                "legacy_compat_abs_sum"
            ]
            / aggregate["element_count"],
            "legacy_compat_exact_pixel_fraction": aggregate[
                "legacy_compat_equal_pixels"
            ]
            / aggregate["element_count"],
            "new_zero_fraction": aggregate["new_zero_pixels"]
            / aggregate["element_count"],
            "old_zero_fraction": aggregate["old_zero_pixels"]
            / aggregate["element_count"],
            "new_min": aggregate["new_min"],
            "new_max": aggregate["new_max"],
            "old_min": aggregate["old_min"],
            "old_max": aggregate["old_max"],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
