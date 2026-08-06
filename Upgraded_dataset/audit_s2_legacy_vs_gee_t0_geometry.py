#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, zoom


COMMON_BANDS = (0, 1, 2, 3, 4, 5, 6, 10, 11)
NATIVE_PIXEL_METERS = 10.0


def read_chw(path: str) -> np.ndarray:
    image = np.asarray(tifffile.imread(path), dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected 3D TIFF, got {image.shape}: {path}")
    if image.shape[0] != 12 and image.shape[-1] == 12:
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] != 12:
        raise ValueError(f"expected 12 bands, got {image.shape}: {path}")
    return image


def resize_chw(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[-2:] == (size, size):
        return image
    factors = (1.0, size / image.shape[-2], size / image.shape[-1])
    resized = zoom(
        image,
        factors,
        order=1,
        mode="nearest",
        prefilter=False,
        grid_mode=True,
    )
    if resized.shape != (image.shape[0], size, size):
        raise ValueError(f"resize produced {resized.shape}, expected {(image.shape[0], size, size)}")
    return resized.astype(np.float32, copy=False)


def harmonize_legacy_common(image: np.ndarray) -> np.ndarray:
    common = image[list(COMMON_BANDS)].astype(np.float32, copy=True)
    return np.where(np.abs(common) < 0.5, 0.0, common - 1000.0)


def normalize_bands(image: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(image, dtype=np.float32)
    for index, band in enumerate(image):
        valid = np.isfinite(band) & (band > 0)
        if valid.sum() < 32:
            continue
        values = band[valid]
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) * 1.4826)
        if scale < 1e-6:
            scale = float(values.std())
        if scale < 1e-6:
            continue
        normalized[index] = (band - center) / scale
    return normalized


def shift_slices(size: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(shift, size), slice(0, size - shift)
    return slice(0, size + shift), slice(-shift, size)


def overlap_slices(
    height: int,
    width: int,
    shift_y: int,
    shift_x: int,
) -> tuple[slice, slice, slice, slice]:
    old_y, new_y = shift_slices(height, shift_y)
    old_x, new_x = shift_slices(width, shift_x)
    return old_y, old_x, new_y, new_x


def pearson(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 64:
        return float("nan")
    x = left[valid].astype(np.float64)
    y = right[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def composite(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.any(normalized != 0, axis=0)
    value = np.median(normalized, axis=0)
    value = value - gaussian_filter(value, sigma=1.0, mode="nearest")
    value[~valid] = 0.0
    return value, valid


def alignment_score(
    old_composite: np.ndarray,
    new_composite: np.ndarray,
    old_valid: np.ndarray,
    new_valid: np.ndarray,
    shift_y: int,
    shift_x: int,
) -> float:
    old_y, old_x, new_y, new_x = overlap_slices(
        old_composite.shape[0],
        old_composite.shape[1],
        shift_y,
        shift_x,
    )
    valid = old_valid[old_y, old_x] & new_valid[new_y, new_x]
    return pearson(
        old_composite[old_y, old_x],
        new_composite[new_y, new_x],
        valid,
    )


def parabolic_offset(minus: float, center: float, plus: float) -> float:
    if not all(np.isfinite([minus, center, plus])):
        return 0.0
    denominator = minus - 2.0 * center + plus
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (minus - plus) / denominator, -1.0, 1.0))


def best_shift(
    old_normalized: np.ndarray,
    new_normalized: np.ndarray,
    radius: int,
) -> tuple[int, int, float, float, dict[tuple[int, int], float]]:
    old_composite, old_valid = composite(old_normalized)
    new_composite, new_valid = composite(new_normalized)
    scores = {}
    best = (float("-inf"), 0, 0)
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            score = alignment_score(
                old_composite,
                new_composite,
                old_valid,
                new_valid,
                shift_y,
                shift_x,
            )
            scores[(shift_y, shift_x)] = score
            candidate = (score if np.isfinite(score) else float("-inf"), -abs(shift_y) - abs(shift_x), -shift_y, -shift_x)
            incumbent = (best[0], -abs(best[1]) - abs(best[2]), -best[1], -best[2])
            if candidate > incumbent:
                best = (candidate[0], shift_y, shift_x)
    score, shift_y, shift_x = best
    if not np.isfinite(score):
        raise ValueError("no finite alignment score")
    subpixel_y = float(shift_y)
    subpixel_x = float(shift_x)
    if -radius < shift_y < radius:
        subpixel_y += parabolic_offset(
            scores[(shift_y - 1, shift_x)],
            scores[(shift_y, shift_x)],
            scores[(shift_y + 1, shift_x)],
        )
    if -radius < shift_x < radius:
        subpixel_x += parabolic_offset(
            scores[(shift_y, shift_x - 1)],
            scores[(shift_y, shift_x)],
            scores[(shift_y, shift_x + 1)],
        )
    return shift_y, shift_x, subpixel_y, subpixel_x, scores


def residual_metrics(
    old_dn: np.ndarray,
    new_dn: np.ndarray,
    old_normalized: np.ndarray,
    new_normalized: np.ndarray,
    shift_y: int,
    shift_x: int,
) -> dict[str, float]:
    old_y, old_x, new_y, new_x = overlap_slices(
        old_dn.shape[-2],
        old_dn.shape[-1],
        shift_y,
        shift_x,
    )
    old_values = old_dn[:, old_y, old_x]
    new_values = new_dn[:, new_y, new_x]
    old_z = old_normalized[:, old_y, old_x]
    new_z = new_normalized[:, new_y, new_x]
    valid = (
        np.isfinite(old_values)
        & np.isfinite(new_values)
        & (old_values > 0)
        & (new_values > 0)
    )
    if valid.sum() < 256:
        raise ValueError("insufficient valid overlap")
    delta_dn = old_values[valid] - new_values[valid]
    delta_z = old_z[valid] - new_z[valid]
    band_correlations = []
    affine_nrmse = []
    for band in range(old_values.shape[0]):
        band_valid = valid[band]
        if band_valid.sum() < 64:
            continue
        old_band = old_values[band][band_valid].astype(np.float64)
        new_band = new_values[band][band_valid].astype(np.float64)
        correlation = np.corrcoef(old_band, new_band)[0, 1]
        slope, intercept = np.polyfit(new_band, old_band, 1)
        residual = old_band - (new_band * slope + intercept)
        scale = max(float(old_band.std()), 1e-6)
        band_correlations.append(float(correlation))
        affine_nrmse.append(float(np.sqrt(np.mean(np.square(residual))) / scale))
    overlap_elements = valid.size
    return {
        "valid_fraction": float(valid.sum() / overlap_elements),
        "mae_dn": float(np.mean(np.abs(delta_dn))),
        "rmse_dn": float(np.sqrt(np.mean(np.square(delta_dn)))),
        "mae_robust_z": float(np.mean(np.abs(delta_z))),
        "rmse_robust_z": float(np.sqrt(np.mean(np.square(delta_z)))),
        "median_band_correlation": float(np.nanmedian(band_correlations)),
        "median_affine_nrmse": float(np.nanmedian(affine_nrmse)),
    }


def process_pair(item: dict[str, object], radius: int, grid_size: int) -> dict[str, object]:
    old_raw = read_chw(str(item["old_t0_path"]))
    new_raw = read_chw(str(item["new_t0_path"]))
    old_resized = resize_chw(old_raw, grid_size)
    new_resized = resize_chw(new_raw, grid_size)
    old_common = harmonize_legacy_common(old_resized)
    new_common = new_resized[list(COMMON_BANDS)].astype(np.float32, copy=False)
    old_normalized = normalize_bands(old_common)
    new_normalized = normalize_bands(new_common)
    shift_y, shift_x, subpixel_y, subpixel_x, scores = best_shift(
        old_normalized,
        new_normalized,
        radius,
    )
    before = residual_metrics(
        old_common,
        new_common,
        old_normalized,
        new_normalized,
        0,
        0,
    )
    after = residual_metrics(
        old_common,
        new_common,
        old_normalized,
        new_normalized,
        shift_y,
        shift_x,
    )
    return {
        **item,
        "legacy_shape": "x".join(map(str, old_raw.shape)),
        "new_gee_shape": "x".join(map(str, new_raw.shape)),
        "grid_size": grid_size,
        "shift_y_to_apply_new_px": shift_y,
        "shift_x_to_apply_new_px": shift_x,
        "subpixel_shift_y_to_apply_new_px": subpixel_y,
        "subpixel_shift_x_to_apply_new_px": subpixel_x,
        "subpixel_shift_magnitude_px": float(math.hypot(subpixel_y, subpixel_x)),
        "subpixel_shift_magnitude_m": float(
            math.hypot(subpixel_y, subpixel_x) * NATIVE_PIXEL_METERS
        ),
        "search_peak_correlation": float(scores[(shift_y, shift_x)]),
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
        "mae_dn_ratio_after_before": float(after["mae_dn"] / max(before["mae_dn"], 1e-12)),
        "rmse_robust_z_ratio_after_before": float(
            after["rmse_robust_z"] / max(before["rmse_robust_z"], 1e-12)
        ),
    }


def quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cohort-csv", required=True)
    parser.add_argument("--new-gee-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=36)
    parser.add_argument("--max-plumes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--max-failure-fraction", type=float, default=0.01)
    args = parser.parse_args()

    old = pd.read_csv(
        args.legacy_cohort_csv,
        usecols=["id", "s2_0_path"],
        low_memory=False,
    ).rename(columns={"s2_0_path": "old_t0_path"})
    new = pd.read_csv(
        args.new_gee_csv,
        usecols=[
            "id",
            "plume_id",
            "label",
            "dx_anchor_px",
            "dy_anchor_px",
            "s2_0_path",
        ],
        low_memory=False,
    ).rename(columns={"s2_0_path": "new_t0_path"})
    if old["id"].duplicated().any() or new["id"].duplicated().any():
        raise ValueError("id must be unique in both manifests")
    frame = new.merge(old, on="id", how="left", validate="one_to_one")
    if frame["old_t0_path"].isna().any():
        raise ValueError("some new GEE rows have no legacy t0 counterpart")
    frame["anchor_radius_sq"] = (
        pd.to_numeric(frame["dx_anchor_px"], errors="raise").pow(2)
        + pd.to_numeric(frame["dy_anchor_px"], errors="raise").pow(2)
    )
    representatives = (
        frame.sort_values(["plume_id", "anchor_radius_sq", "id"])
        .drop_duplicates("plume_id")
        .reset_index(drop=True)
    )
    if args.max_plumes > 0 and len(representatives) > args.max_plumes:
        representatives = representatives.sample(
            n=args.max_plumes,
            random_state=args.seed,
        ).sort_values("plume_id")
    items = representatives[
        [
            "id",
            "plume_id",
            "label",
            "dx_anchor_px",
            "dy_anchor_px",
            "anchor_radius_sq",
            "old_t0_path",
            "new_t0_path",
        ]
    ].to_dict("records")

    records = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pair, item, args.search_radius, args.grid_size): item
            for item in items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "id": item["id"],
                        "plume_id": item["plume_id"],
                        "error": repr(exc),
                    }
                )
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"[Geometry] {completed}/{len(futures)} failures={len(failures)}",
                    flush=True,
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_frame = pd.DataFrame(records).sort_values("plume_id")
    records_frame.to_csv(output_dir / "per_plume_geometry.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    failure_fraction = len(failures) / max(len(items), 1)
    summary = {
        "legacy_cohort_csv": args.legacy_cohort_csv,
        "new_gee_csv": args.new_gee_csv,
        "candidate_plumes": int(frame["plume_id"].nunique()),
        "selected_plumes": len(items),
        "successful_plumes": len(records),
        "failed_plumes": len(failures),
        "failure_fraction": failure_fraction,
        "representative_rule": "minimum dx_anchor_px^2 + dy_anchor_px^2, then minimum id",
        "common_bands": COMMON_BANDS,
        "legacy_radiometric_harmonization": "subtract 1000 DN from nonzero common bands",
        "registration_grid": f"{args.grid_size}x{args.grid_size}",
        "native_pixel_meters": NATIVE_PIXEL_METERS,
        "search_radius_px": args.search_radius,
        "shift_semantics": "shift to apply to new GEE image to align with legacy image",
        "signed_shift_y_px": quantiles(
            records_frame["subpixel_shift_y_to_apply_new_px"]
        ),
        "signed_shift_x_px": quantiles(
            records_frame["subpixel_shift_x_to_apply_new_px"]
        ),
        "shift_magnitude_px": quantiles(records_frame["subpixel_shift_magnitude_px"]),
        "shift_magnitude_m": quantiles(records_frame["subpixel_shift_magnitude_m"]),
        "shift_fraction": {
            "integer_zero": float(
                (
                    records_frame["shift_x_to_apply_new_px"].eq(0)
                    & records_frame["shift_y_to_apply_new_px"].eq(0)
                ).mean()
            ),
            "subpixel_le_1px": float(
                records_frame["subpixel_shift_magnitude_px"].le(1.0).mean()
            ),
            "subpixel_le_2px": float(
                records_frame["subpixel_shift_magnitude_px"].le(2.0).mean()
            ),
            "subpixel_gt_4px": float(
                records_frame["subpixel_shift_magnitude_px"].gt(4.0).mean()
            ),
            "search_boundary": float(
                (
                    records_frame["shift_x_to_apply_new_px"].abs().eq(args.search_radius)
                    | records_frame["shift_y_to_apply_new_px"].abs().eq(args.search_radius)
                ).mean()
            ),
        },
        "before_after": {
            "mae_dn_improved_fraction": float(
                records_frame["mae_dn_ratio_after_before"].lt(1.0).mean()
            ),
            "rmse_robust_z_improved_fraction": float(
                records_frame["rmse_robust_z_ratio_after_before"].lt(1.0).mean()
            ),
            "median_band_correlation_improved_fraction": float(
                (
                    records_frame["after_median_band_correlation"]
                    > records_frame["before_median_band_correlation"]
                ).mean()
            ),
            "median_band_correlation_before": quantiles(
                records_frame["before_median_band_correlation"]
            ),
            "median_band_correlation_after": quantiles(
                records_frame["after_median_band_correlation"]
            ),
            "mae_dn_before": quantiles(records_frame["before_mae_dn"]),
            "mae_dn_after": quantiles(records_frame["after_mae_dn"]),
            "mae_dn_ratio_after_before": quantiles(
                records_frame["mae_dn_ratio_after_before"]
            ),
            "rmse_robust_z_before": quantiles(
                records_frame["before_rmse_robust_z"]
            ),
            "rmse_robust_z_after": quantiles(records_frame["after_rmse_robust_z"]),
            "rmse_robust_z_ratio_after_before": quantiles(
                records_frame["rmse_robust_z_ratio_after_before"]
            ),
            "median_affine_nrmse_before": quantiles(
                records_frame["before_median_affine_nrmse"]
            ),
            "median_affine_nrmse_after": quantiles(
                records_frame["after_median_affine_nrmse"]
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failure_fraction > args.max_failure_fraction:
        raise RuntimeError(
            f"failure fraction {failure_fraction:.4f} exceeds {args.max_failure_fraction:.4f}"
        )


if __name__ == "__main__":
    main()
