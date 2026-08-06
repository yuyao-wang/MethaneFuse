#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import tifffile


TIMEPOINT_FILES = {
    "s2_0_path": ("t0", "s2_0.tif"),
    "s2_90_path": ("seasonal", "s2_seasonal.tif"),
    "s2_360_path": ("year", "s2_year.tif"),
}
OUTPUT_NAMES = {
    "s2_0_path": "s2_0.tif",
    "s2_90_path": "s2_90.tif",
    "s2_360_path": "s2_360.tif",
}
S2_BAND_INDEX = 11
S2_ZERO_RATIO_THRESH = 0.20
WRITE_LOCK = threading.Lock()


def to_chw(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"expected a 3D S2 TIFF, got shape={data.shape}")
    if data.shape[0] <= 32:
        return data
    if data.shape[-1] <= 32:
        return data.transpose(2, 0, 1)
    raise ValueError(f"cannot determine S2 band axis from shape={data.shape}")


def raw_path(raw_root: Path, plume_id: str, timepoint: str, name: str) -> Path:
    return raw_root / "S2_GEE_6time" / timepoint / plume_id / name


def valid_crop(crops: list[np.ndarray]) -> bool:
    for crop in crops:
        if crop.shape[0] <= S2_BAND_INDEX:
            return False
        if float((crop[S2_BAND_INDEX] == 0).mean()) >= S2_ZERO_RATIO_THRESH:
            return False
    return True


def read_bhw(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        image = dataset.read()
    if image.shape[0] == 1:
        stacked = np.asarray(tifffile.imread(path))
        if stacked.ndim == 3:
            image = to_chw(stacked)
    return image


def alignment_shift(
    raw_image: np.ndarray,
    reference_path: Path,
    image_size: int,
    search_radius: int,
) -> tuple[int, int, float, float]:
    reference = read_bhw(reference_path).astype(np.float32)
    if reference.shape != (12, image_size, image_size):
        raise ValueError(f"unexpected reference shape={reference.shape}")
    _, height, width = raw_image.shape
    base_top = (height - image_size) // 2
    base_left = (width - image_size) // 2
    best: tuple[float, float, int, int] | None = None
    reference_sample = reference[:, ::8, ::8]
    for shift_y in range(-search_radius, search_radius + 1):
        for shift_x in range(-search_radius, search_radius + 1):
            top = base_top + shift_y
            left = base_left + shift_x
            if (
                top < 0
                or left < 0
                or top + image_size > height
                or left + image_size > width
            ):
                continue
            candidate = raw_image[
                :,
                top : top + image_size,
                left : left + image_size,
            ].astype(np.float32)
            candidate_sample = candidate[:, ::8, ::8]
            mae = float(np.abs(candidate_sample - reference_sample).mean())
            correlation = float(
                np.corrcoef(
                    candidate_sample.ravel(),
                    reference_sample.ravel(),
                )[0, 1]
            )
            score = (mae, -correlation, shift_y, shift_x)
            if best is None or score < best:
                best = score
    if best is None:
        raise ValueError("no valid alignment candidate")
    return best[2], best[3], best[0], -best[1]


def atomic_tiff_write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.tif")
    tifffile.imwrite(temporary, data)
    os.replace(temporary, path)


def existing_outputs(output_root: Path, sample_id: int) -> dict[str, str] | None:
    sample_dir = output_root / f"group_{sample_id:08d}"
    paths = {
        column: str(sample_dir / name)
        for column, name in OUTPUT_NAMES.items()
    }
    if all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths.values()):
        return paths
    return None


def process_plume(
    plume_id: str,
    rows: pd.DataFrame,
    raw_root: Path,
    output_root: Path,
    crop_size: int,
    image_size: int,
    alignment_references: dict[str, dict[str, str]],
    alignment_search_radius: int,
    alignment_min_correlation: float,
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sources = {
        column: raw_path(raw_root, plume_id, timepoint, name)
        for column, (timepoint, name) in TIMEPOINT_FILES.items()
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        return [], {
            "plume_id": plume_id,
            "status": "missing_raw",
            "missing": missing,
            "rows": int(len(rows)),
        }

    try:
        images = {
            column: read_bhw(path)
            for column, path in sources.items()
        }
    except Exception as exc:
        return [], {
            "plume_id": plume_id,
            "status": "read_error",
            "message": f"{type(exc).__name__}: {exc}",
            "rows": int(len(rows)),
        }

    reference_record = alignment_references.get(plume_id, {})
    alignment_candidates = [
        ("seasonal", "s2_90_path", reference_record.get("seasonal", "")),
        ("year", "s2_360_path", reference_record.get("year", "")),
    ]
    alignment = None
    alignment_errors = []
    for reference_name, raw_column, reference_value in alignment_candidates:
        reference_path = Path(reference_value) if reference_value else None
        if reference_path is None or not reference_path.is_file():
            alignment_errors.append(f"{reference_name}:missing_reference")
            continue
        try:
            shift_y, shift_x, mae, correlation = alignment_shift(
                images[raw_column],
                reference_path,
                image_size=image_size,
                search_radius=alignment_search_radius,
            )
        except Exception as exc:
            alignment_errors.append(
                f"{reference_name}:{type(exc).__name__}:{exc}"
            )
            continue
        if correlation < alignment_min_correlation:
            alignment_errors.append(
                f"{reference_name}:corr={correlation:.6f},mae={mae:.6f}"
            )
            continue
        alignment = {
            "reference": reference_name,
            "reference_path": str(reference_path),
            "shift_y": shift_y,
            "shift_x": shift_x,
            "mae_sampled": mae,
            "correlation_sampled": correlation,
        }
        break
    if alignment is None:
        return [], {
            "plume_id": plume_id,
            "status": "alignment_failed",
            "alignment_errors": alignment_errors,
            "rows": int(len(rows)),
        }

    standardized = {}
    source_shapes = {}
    for column, image in images.items():
        channels, height, width = image.shape
        source_shapes[column] = [channels, height, width]
        if channels != 12 or height < image_size or width < image_size:
            return [], {
                "plume_id": plume_id,
                "status": "unexpected_shape",
                "shapes": source_shapes,
                "rows": int(len(rows)),
            }
        top = (height - image_size) // 2 + int(alignment["shift_y"])
        left = (width - image_size) // 2 + int(alignment["shift_x"])
        if (
            top < 0
            or left < 0
            or top + image_size > height
            or left + image_size > width
        ):
            return [], {
                "plume_id": plume_id,
                "status": "aligned_window_out_of_bounds",
                "alignment": alignment,
                "shapes": source_shapes,
                "rows": int(len(rows)),
            }
        standardized[column] = image[
            :,
            top : top + image_size,
            left : left + image_size,
        ].astype(np.float32)
    images = standardized
    height = width = image_size

    center = image_size // 2
    half = crop_size // 2
    output_rows: list[dict[str, object]] = []
    quality_failed = 0
    out_of_bounds = 0
    resumed = 0
    for row in rows.to_dict("records"):
        sample_id = int(row["id"])
        if resume:
            paths = existing_outputs(output_root, sample_id)
            if paths is not None:
                row.update(paths)
                output_rows.append(row)
                resumed += 1
                continue

        left = center + int(row["dx_anchor_px"]) - half
        top = center + int(row["dy_anchor_px"]) - half
        if left < 0 or top < 0 or left + crop_size > width or top + crop_size > height:
            out_of_bounds += 1
            continue
        crops = {
            column: image[:, top : top + crop_size, left : left + crop_size]
            for column, image in images.items()
        }
        if not valid_crop(list(crops.values())):
            quality_failed += 1
            continue

        sample_dir = output_root / f"group_{sample_id:08d}"
        written = {}
        for column, crop in crops.items():
            output_path = sample_dir / OUTPUT_NAMES[column]
            atomic_tiff_write(output_path, crop)
            written[column] = str(output_path)
        row.update(written)
        output_rows.append(row)

    return output_rows, {
        "plume_id": plume_id,
        "status": "ok",
        "rows": int(len(rows)),
        "written_rows": int(len(output_rows)),
        "quality_failed_rows": int(quality_failed),
        "out_of_bounds_rows": int(out_of_bounds),
        "resumed_rows": int(resumed),
        "source_shapes": source_shapes,
        "alignment": alignment,
    }


def remap_split(source_csv: Path, path_map: pd.DataFrame, output_csv: Path) -> dict[str, object]:
    source = pd.read_csv(source_csv, low_memory=False)
    path_columns = ["id", *TIMEPOINT_FILES]
    merged = source.drop(columns=list(TIMEPOINT_FILES), errors="ignore").merge(
        path_map[path_columns],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return {
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "source_rows": int(len(source)),
        "output_rows": int(len(merged)),
        "dropped_rows": int(len(source) - len(merged)),
        "labels": {
            str(key): int(value)
            for key, value in merged["label"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--alignment-reference-csv", required=True)
    parser.add_argument(
        "--legacy-reference-root",
        default="/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/Dataset/plume_raw_s2_gee90360_512",
    )
    parser.add_argument("--split-root")
    parser.add_argument("--split-output-root")
    parser.add_argument("--crop-size", type=int, default=36)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--alignment-search-radius", type=int, default=3)
    parser.add_argument("--alignment-min-correlation", type=float, default=0.995)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-plumes", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    source = pd.read_csv(args.source_csv, low_memory=False)
    required = {
        "id",
        "plume_id",
        "label",
        "dx_anchor_px",
        "dy_anchor_px",
    }
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise ValueError(f"missing source columns: {missing_columns}")
    if source["id"].duplicated().any():
        raise ValueError("source IDs are not unique")
    if args.limit_plumes > 0:
        selected_plumes = source["plume_id"].drop_duplicates().head(args.limit_plumes)
        source = source[source["plume_id"].isin(set(selected_plumes))].copy()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    alignment_frame = pd.read_csv(args.alignment_reference_csv, low_memory=False)
    alignment_required = {
        "plume_id",
        "legacy_seasonal_512_path",
        "legacy_year_512_path",
    }
    missing_alignment_columns = sorted(
        alignment_required - set(alignment_frame.columns)
    )
    if missing_alignment_columns:
        raise ValueError(
            f"missing alignment columns: {missing_alignment_columns}"
        )
    legacy_reference_root = Path(args.legacy_reference_root)
    alignment_references = {}
    for row in alignment_frame[
        [
            "plume_id",
            "legacy_seasonal_512_path",
            "legacy_year_512_path",
        ]
    ].itertuples(index=False):
        plume_id = str(row.plume_id)
        seasonal = Path(str(row.legacy_seasonal_512_path))
        year = Path(str(row.legacy_year_512_path))
        if not seasonal.is_file():
            seasonal = legacy_reference_root / plume_id / "s2_90_std_512.tif"
        if not year.is_file():
            year = legacy_reference_root / plume_id / "s2_360_std_512.tif"
        alignment_references[plume_id] = {
            "seasonal": str(seasonal),
            "year": str(year),
        }
    grouped = list(source.groupby("plume_id", sort=True))
    records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                process_plume,
                str(plume_id),
                rows.copy(),
                raw_root,
                output_root,
                args.crop_size,
                args.image_size,
                alignment_references,
                args.alignment_search_radius,
                args.alignment_min_correlation,
                args.resume,
            ): str(plume_id)
            for plume_id, rows in grouped
        }
        for index, future in enumerate(as_completed(futures), start=1):
            plume_rows, diagnostic = future.result()
            with WRITE_LOCK:
                records.extend(plume_rows)
                diagnostics.append(diagnostic)
            if index % max(1, args.progress_every) == 0 or index == len(futures):
                print(
                    f"[Crop] plumes={index}/{len(futures)} rows={len(records)}/{len(source)}",
                    flush=True,
                )

    output = pd.DataFrame(records)
    if not output.empty:
        output = output.sort_values("id").reset_index(drop=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    status_counts = pd.Series(
        [record["status"] for record in diagnostics], dtype="string"
    ).value_counts().to_dict()
    report: dict[str, object] = {
        "source_csv": args.source_csv,
        "raw_root": args.raw_root,
        "output_root": args.output_root,
        "output_csv": args.output_csv,
        "alignment_reference_csv": args.alignment_reference_csv,
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "dropped_rows": int(len(source) - len(output)),
        "source_plumes": int(source["plume_id"].nunique()),
        "output_plumes": int(output["plume_id"].nunique()) if not output.empty else 0,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "quality_failed_rows": int(
            sum(record.get("quality_failed_rows", 0) for record in diagnostics)
        ),
        "out_of_bounds_rows": int(
            sum(record.get("out_of_bounds_rows", 0) for record in diagnostics)
        ),
    }

    if args.split_root and args.split_output_root:
        split_reports = {}
        for split_name in ("row_random_80_20", "event_disjoint_80_20"):
            for partition in ("train", "test"):
                key = f"{split_name}/{partition}"
                split_reports[key] = remap_split(
                    Path(args.split_root) / split_name / f"{partition}.csv",
                    output,
                    Path(args.split_output_root) / split_name / f"{partition}.csv",
                )
        report["splits"] = split_reports

    report_path = output_csv.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    diagnostics_path = output_csv.with_suffix(".plume_diagnostics.jsonl")
    diagnostics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in diagnostics)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
