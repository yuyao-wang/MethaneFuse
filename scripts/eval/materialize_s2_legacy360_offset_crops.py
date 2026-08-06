#!/usr/bin/env python3
"""Materialize or verify legacy-360 S2 crops with one shared offset recipe.

Both the within-scene distance dataset and the original-table dataset use the
same operation:

1. resolve the three raw temporal images for an event;
2. take a 36 x 36 crop at raw-image centre plus ``dx_anchor_px/dy_anchor_px``;
3. resize it to 224 x 224 with an explicitly selected interpolation contract.

The source-audit JSON is required because it is the sealed mapping from each
event to its exact t0, seasonal, and year raw images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F


SLOTS = (
    ("t0", "t0_raw_path", "s2_0_path", "s2_0.tif"),
    ("seasonal", "seasonal_raw_path", "s2_90_path", "s2_90.tif"),
    ("year", "year_raw_path", "s2_360_path", "s2_360.tif"),
)
T0_ZERO_CHANNELS = (8, 9)
EXPECTED_CHANNELS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("materialize", "verify"), required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-audit-json", type=Path, required=True)
    parser.add_argument("--crop-pixels", type=int, default=36)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument(
        "--resize-contract",
        choices=("torch_align_corners_false", "legacy_endpoint"),
        default="legacy_endpoint",
        help=(
            "Interpolation geometry. The historical training generator used "
            "torch_align_corners_false; legacy_endpoint preserves the first "
            "version of this evaluation materializer."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "materialize":
        if args.output_root is None or args.output_manifest is None:
            parser.error(
                "materialize mode requires --output-root and --output-manifest"
            )
    elif args.reference_manifest is None:
        parser.error("verify mode requires --reference-manifest")
    if args.crop_pixels <= 0 or args.crop_pixels % 2:
        parser.error("--crop-pixels must be a positive even integer")
    if args.output_size <= 0:
        parser.error("--output-size must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def safe_component(value: Any) -> str:
    text = clean_text(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return safe[:220] if safe else "missing"


def read_chw(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(tifffile.imread(path))
    if image.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D TIFF, got {image.shape}")
    if image.shape[0] == EXPECTED_CHANNELS:
        return image
    if image.shape[-1] == EXPECTED_CHANNELS:
        return image.transpose(2, 0, 1)
    raise ValueError(f"{path}: expected 12 channels, got {image.shape}")


def legacy_endpoint_resize(crop: np.ndarray, output_size: int) -> np.ndarray:
    """Match the endpoint-aligned NumPy bilinear implementation exactly."""
    channels, height, width = crop.shape
    ys = np.linspace(0, height - 1, output_size)
    xs = np.linspace(0, width - 1, output_size)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    output = np.empty((channels, output_size, output_size), dtype=np.float32)
    for band_index in range(channels):
        band = crop[band_index]
        a = band[y0[:, None], x0[None, :]]
        b = band[y0[:, None], x1[None, :]]
        c = band[y1[:, None], x0[None, :]]
        d = band[y1[:, None], x1[None, :]]
        output[band_index] = (
            a * (1 - wx) * (1 - wy)
            + b * wx * (1 - wy)
            + c * (1 - wx) * wy
            + d * wx * wy
        )
    return output


def torch_align_corners_false_resize(
    crop: np.ndarray, output_size: int
) -> np.ndarray:
    """Match the historical generator's F.interpolate call exactly."""
    tensor = torch.from_numpy(
        np.ascontiguousarray(crop.astype(np.float32, copy=False))
    ).unsqueeze(0)
    with torch.inference_mode():
        output = F.interpolate(
            tensor,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
    return output.squeeze(0).cpu().numpy()


def offset_crop(
    image: np.ndarray,
    *,
    dx: int,
    dy: int,
    crop_pixels: int,
    output_size: int,
    resize_contract: str,
) -> np.ndarray:
    _, height, width = image.shape
    half = crop_pixels // 2
    top = height // 2 + int(dy) - half
    left = width // 2 + int(dx) - half
    bottom = top + crop_pixels
    right = left + crop_pixels
    if top < 0 or left < 0 or bottom > height or right > width:
        raise ValueError(
            "offset crop is outside the raw image: "
            f"shape={image.shape}, dx={dx}, dy={dy}, "
            f"window=({top}:{bottom}, {left}:{right})"
        )
    crop = image[:, top:bottom, left:right].astype(np.float32, copy=False)
    if crop.shape != (EXPECTED_CHANNELS, crop_pixels, crop_pixels):
        raise ValueError(f"unexpected crop shape: {crop.shape}")
    if resize_contract == "torch_align_corners_false":
        return torch_align_corners_false_resize(crop, output_size)
    if resize_contract == "legacy_endpoint":
        return legacy_endpoint_resize(crop, output_size)
    raise ValueError(f"unsupported resize contract: {resize_contract}")


def validate_temporal_layout(image: np.ndarray, slot: str, context: str) -> None:
    if image.dtype != np.float32:
        raise ValueError(f"{context}: expected float32, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"{context}: non-finite values")
    all_zero = [index for index in range(EXPECTED_CHANNELS) if not np.any(image[index])]
    if slot == "t0":
        if all_zero != list(T0_ZERO_CHANNELS):
            raise ValueError(
                f"{context}: t0 zero channels={all_zero}, "
                f"expected={list(T0_ZERO_CHANNELS)}"
            )
    elif all_zero:
        raise ValueError(f"{context}: historical all-zero channels={all_zero}")
    b12_zero_ratio = float((image[11] == 0).mean())
    if b12_zero_ratio >= 0.20:
        raise ValueError(f"{context}: B12 zero ratio={b12_zero_ratio:.6f}")


def atomic_tiff_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.tif")
    tifffile.imwrite(temporary, image)
    os.replace(temporary, path)


def load_event_sources(audit_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("pass") is not True:
        raise ValueError(f"source audit did not pass: {audit_path}")
    raw_sources = audit.get("event_sources")
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError(f"source audit has no event_sources: {audit_path}")
    sources = {str(key): dict(value) for key, value in raw_sources.items()}
    for event_key, record in sources.items():
        for _, raw_column, _, _ in SLOTS:
            value = clean_text(record.get(raw_column))
            if not value:
                raise ValueError(f"{event_key}: missing {raw_column}")
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
    return sources, audit


def build_event_resolver(
    event_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    by_truth: dict[str, str] = {}
    ambiguous: set[str] = set()
    for event_key, record in event_sources.items():
        truth_id = clean_text(record.get("matched_ground_truth_sample_id"))
        if not truth_id:
            continue
        if truth_id in by_truth and by_truth[truth_id] != event_key:
            ambiguous.add(truth_id)
        else:
            by_truth[truth_id] = event_key
    for truth_id in ambiguous:
        by_truth.pop(truth_id, None)
    by_prefixed_id = {f"s2_{event_key}": event_key for event_key in event_sources}
    return by_truth, by_prefixed_id


def resolve_event_key(
    row: Mapping[str, Any],
    event_sources: Mapping[str, Mapping[str, Any]],
    by_truth: Mapping[str, str],
    by_prefixed_id: Mapping[str, str],
) -> str:
    plume_id = clean_text(row.get("plume_id"))
    row_id = clean_text(row.get("id"))
    candidates = (
        plume_id if plume_id in event_sources else "",
        row_id if row_id in event_sources else "",
        by_prefixed_id.get(row_id, ""),
        by_truth.get(plume_id, ""),
    )
    resolved = [value for value in candidates if value]
    if not resolved:
        raise KeyError(f"cannot resolve event for id={row_id}, plume_id={plume_id}")
    if len(set(resolved)) != 1:
        raise ValueError(
            f"ambiguous event resolution for id={row_id}, plume_id={plume_id}: "
            f"{sorted(set(resolved))}"
        )
    return resolved[0]


def validate_manifest(frame: pd.DataFrame, path: Path) -> None:
    required = {"id", "plume_id", "label", "dx_anchor_px", "dy_anchor_px"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns={missing}")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate id values")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not labels.isin((0, 1)).all():
        raise ValueError(f"{path}: labels must be 0/1")
    for column in ("dx_anchor_px", "dy_anchor_px"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(values, np.round(values)).all():
            raise ValueError(f"{path}: non-integer values in {column}")


def source_images(
    event_key: str,
    event_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    record = event_sources[event_key]
    images = {}
    for slot, raw_column, _, _ in SLOTS:
        images[slot] = read_chw(Path(str(record[raw_column])))
    return images


def process_event(
    event_key: str,
    rows: list[dict[str, Any]],
    *,
    event_sources: Mapping[str, Mapping[str, Any]],
    crop_pixels: int,
    output_size: int,
    resize_contract: str,
    mode: str,
    output_root: Path | None,
    reference_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = source_images(event_key, event_sources)
    output_rows: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    files_checked = 0
    files_written = 0

    for source_row in rows:
        row = dict(source_row)
        row_id = clean_text(row["id"])
        dx = int(row["dx_anchor_px"])
        dy = int(row["dy_anchor_px"])
        reference_row = reference_by_id.get(row_id)
        for slot, _, output_column, output_name in SLOTS:
            chip = offset_crop(
                images[slot],
                dx=dx,
                dy=dy,
                crop_pixels=crop_pixels,
                output_size=output_size,
                resize_contract=resize_contract,
            )
            validate_temporal_layout(chip, slot, f"id={row_id}/{slot}")
            if mode == "verify":
                if reference_row is None:
                    raise KeyError(f"reference manifest has no id={row_id}")
                reference_path = Path(str(reference_row[output_column]))
                reference = read_chw(reference_path).astype(np.float32, copy=False)
                difference = float(np.max(np.abs(chip - reference)))
                max_abs_diff = max(max_abs_diff, difference)
                if difference != 0.0:
                    raise ValueError(
                        f"id={row_id}/{slot}: shared materializer differs from "
                        f"{reference_path}; max_abs_diff={difference}"
                    )
                files_checked += 1
            else:
                assert output_root is not None
                output_path = (
                    output_root.resolve()
                    / "crops"
                    / safe_component(row_id)
                    / output_name
                )
                if output_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite existing output: {output_path}"
                    )
                atomic_tiff_write(output_path, chip)
                row[output_column] = str(output_path)
                files_written += 1
        output_rows.append(row)

    return output_rows, {
        "event_key": event_key,
        "rows": len(rows),
        "files_checked": files_checked,
        "files_written": files_written,
        "max_abs_diff": max_abs_diff,
        "raw_shapes": {
            slot: list(map(int, image.shape)) for slot, image in images.items()
        },
        "raw_dtypes": {slot: str(image.dtype) for slot, image in images.items()},
    }


def main() -> None:
    args = parse_args()
    source_path = args.source_manifest.resolve()
    source = pd.read_csv(source_path, low_memory=False)
    validate_manifest(source, source_path)
    event_sources, source_audit = load_event_sources(
        args.source_audit_json.resolve()
    )
    by_truth, by_prefixed_id = build_event_resolver(event_sources)

    records = source.to_dict(orient="records")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        event_key = resolve_event_key(
            row, event_sources, by_truth, by_prefixed_id
        )
        grouped.setdefault(event_key, []).append(row)

    reference_by_id: dict[str, dict[str, Any]] = {}
    reference_path = None
    if args.mode == "verify":
        assert args.reference_manifest is not None
        reference_path = args.reference_manifest.resolve()
        reference = pd.read_csv(reference_path, low_memory=False)
        validate_manifest(reference, reference_path)
        reference_by_id = {
            clean_text(row["id"]): row
            for row in reference.to_dict(orient="records")
        }
        source_ids = {clean_text(value) for value in source["id"]}
        if set(reference_by_id) != source_ids:
            raise ValueError(
                "source/reference id sets differ: "
                f"source_only={sorted(source_ids - set(reference_by_id))[:10]}, "
                f"reference_only={sorted(set(reference_by_id) - source_ids)[:10]}"
            )

    output_root = args.output_root.resolve() if args.output_root else None
    if args.mode == "materialize":
        assert output_root is not None
        assert args.output_manifest is not None
        output_manifest = args.output_manifest.resolve()
        if output_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite existing manifest: {output_manifest}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_manifest = None

    output_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                process_event,
                event_key,
                rows,
                event_sources=event_sources,
                crop_pixels=args.crop_pixels,
                output_size=args.output_size,
                resize_contract=args.resize_contract,
                mode=args.mode,
                output_root=output_root,
                reference_by_id=reference_by_id,
            ): event_key
            for event_key, rows in grouped.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            rows, diagnostic = future.result()
            output_records.extend(rows)
            diagnostics.append(diagnostic)
            if index % 10 == 0 or index == len(futures):
                print(
                    f"[{args.mode}] events={index}/{len(futures)} "
                    f"rows={len(output_records)}/{len(source)}",
                    flush=True,
                )

    output_frame = pd.DataFrame(output_records)
    order = {clean_text(value): index for index, value in enumerate(source["id"])}
    output_frame["_source_order"] = output_frame["id"].map(
        lambda value: order[clean_text(value)]
    )
    output_frame = (
        output_frame.sort_values("_source_order")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )
    if args.mode == "materialize":
        assert output_manifest is not None
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(output_manifest, index=False)

    labels = pd.to_numeric(source["label"], errors="raise").astype(int)
    result = {
        "pass": True,
        "mode": args.mode,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "source_audit_json": str(args.source_audit_json.resolve()),
        "source_audit_json_sha256": sha256(args.source_audit_json.resolve()),
        "source_audit_output_manifest": source_audit.get("output_manifest"),
        "reference_manifest": str(reference_path) if reference_path else None,
        "reference_manifest_sha256": (
            sha256(reference_path) if reference_path else None
        ),
        "output_root": str(output_root) if output_root else None,
        "output_manifest": str(output_manifest) if output_manifest else None,
        "output_manifest_sha256": (
            sha256(output_manifest) if output_manifest else None
        ),
        "rows": len(source),
        "events": len(grouped),
        "labels": {
            str(key): int(value)
            for key, value in labels.value_counts().sort_index().items()
        },
        "files_checked": int(
            sum(record["files_checked"] for record in diagnostics)
        ),
        "files_written": int(
            sum(record["files_written"] for record in diagnostics)
        ),
        "max_abs_diff": float(
            max((record["max_abs_diff"] for record in diagnostics), default=0.0)
        ),
        "contract": {
            "crop_origin": "raw image centre plus dx_anchor_px/dy_anchor_px",
            "crop_pixels": int(args.crop_pixels),
            "output_size": int(args.output_size),
            "resize": {
                "torch_align_corners_false": (
                    "torch.nn.functional.interpolate bilinear, "
                    "align_corners=False"
                ),
                "legacy_endpoint": (
                    "legacy endpoint-aligned bilinear (np.linspace)"
                ),
            }[args.resize_contract],
            "stored_dtype": "float32",
            "temporal_order": ["t0", "seasonal/t-90", "year/t-360"],
            "t0_expected_all_zero_channels": list(T0_ZERO_CHANNELS),
            "historical_expected_all_zero_channels": [],
        },
        "event_diagnostics": sorted(
            diagnostics, key=lambda record: record["event_key"]
        ),
    }
    audit_path = args.audit_json.resolve()
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {audit_path}")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "event_diagnostics"
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
