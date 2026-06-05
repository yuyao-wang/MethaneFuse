#!/usr/bin/env python
"""Visualize one MethaneUnion row with optional prediction score and plume-mask overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "MethaneUnion" / "datasets" / "temporal_split" / "480m_GSD" / "test.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "examples" / "prediction_row0.png"

SENSOR_COLUMNS = {
    "s2": ("s2_0_path", "s2_plume_path", "pred_prob1_s2"),
    "l89": ("l89_0_path", "l89_plume_path", "pred_prob1_l89"),
    "emit": ("emit_0_path", "emit_plume_path", "pred_prob1_emit"),
}
RELEASE_RENAMES = {
    "S2_t0_path": "s2_0_path",
    "S2_plume_label_path": "s2_plume_path",
    "L89_t0_path": "l89_0_path",
    "L89_plume_label_path": "l89_plume_path",
    "EMIT_t0_path": "emit_0_path",
    "EMIT_plume_label_path": "emit_plume_path",
}


def valid_path(value: object) -> bool:
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null"}


def read_tiff(path: str) -> np.ndarray:
    arr = np.asarray(tiff.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] == 1:
        rgb = np.repeat(arr[:1], 3, axis=0)
    elif arr.shape[0] == 2:
        rgb = np.concatenate([arr[:2], arr[1:2]], axis=0)
    else:
        rgb = arr[:3]
    rgb = np.moveaxis(rgb, 0, -1)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(rgb)), float(np.nanmax(rgb) + 1e-6)
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def overlay_mask(image: Image.Image, mask_path: str) -> Image.Image:
    mask = np.asarray(tiff.imread(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = mask > 0
    if mask.shape != (image.height, image.width):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, resample=Image.NEAREST)
        mask = np.asarray(mask_img) > 0
    rgba = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay[mask] = [255, 64, 64, 110]
    return Image.alpha_composite(rgba, Image.fromarray(overlay, "RGBA"))


def choose_sensor(row: pd.Series, requested: str) -> str:
    if requested != "auto":
        return requested
    for sensor, (image_col, _, _) in SENSOR_COLUMNS.items():
        if image_col in row and valid_path(row[image_col]):
            return sensor
    raise ValueError("No visualizable S2, L89, or EMIT image path found for this row.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Manifest CSV, optionally with pred_prob1_* columns.")
    parser.add_argument("--index", type=int, default=0, help="Row index to visualize.")
    parser.add_argument("--sensor", choices=["auto", "s2", "l89", "emit"], default="auto")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv).rename(columns=RELEASE_RENAMES)
    row = df.iloc[args.index]
    sensor = choose_sensor(row, args.sensor)
    image_col, mask_col, prob_col = SENSOR_COLUMNS[sensor]

    image = Image.fromarray(to_rgb(read_tiff(str(row[image_col]))))
    if mask_col in row and valid_path(row[mask_col]):
        image = overlay_mask(image, str(row[mask_col]))

    draw = ImageDraw.Draw(image)
    label = row.get("label", "")
    prob = row.get(prob_col, "") if prob_col in row else ""
    text = f"row={args.index} sensor={sensor} label={label}"
    if valid_path(prob):
        text += f" prob1={float(prob):.3f}"
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0, 180))
    draw.text((6, 5), text, fill=(255, 255, 255, 255))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
