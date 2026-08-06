import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


DEFAULT_PATH_COLUMNS = ("s2_0_path", "s2_90_path", "s2_360_path")


def read_moments(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    image = np.asarray(tifffile.imread(path), dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF at {path}, got shape={image.shape}")
    if image.shape[0] != 12 and image.shape[-1] == 12:
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] != 12:
        raise ValueError(f"Expected 12 channels at {path}, got shape={image.shape}")
    return image.sum(axis=(1, 2)), np.square(image).sum(axis=(1, 2)), image.shape[1] * image.shape[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--path-columns", default=",".join(DEFAULT_PATH_COLUMNS))
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    path_columns = [column.strip() for column in args.path_columns.split(",") if column.strip()]
    missing_columns = [column for column in path_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing path columns: {missing_columns}")

    sample_count = min(max(1, args.samples), len(frame))
    sampled = frame.sample(n=sample_count, random_state=args.seed, replace=False)
    paths = sampled[path_columns].to_numpy().reshape(-1).tolist()

    channel_sum = np.zeros(12, dtype=np.float64)
    channel_sumsq = np.zeros(12, dtype=np.float64)
    pixel_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (image_sum, image_sumsq, image_pixels) in enumerate(
            executor.map(read_moments, paths), start=1
        ):
            channel_sum += image_sum
            channel_sumsq += image_sumsq
            pixel_count += image_pixels
            if index % 1000 == 0 or index == len(paths):
                print(f"[Stats] processed {index}/{len(paths)} TIFFs", flush=True)

    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sumsq / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    result = {
        "csv": str(Path(args.csv).resolve()),
        "sampled_rows": sample_count,
        "timepoints": len(path_columns),
        "tiff_count": len(paths),
        "pixel_count_per_channel": pixel_count,
        "seed": args.seed,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
