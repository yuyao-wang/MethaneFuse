#!/usr/bin/env python3
"""Prepare GEE S2 acquisitions for legacy-360 temporal hard-negative tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = {"t0", "seasonal", "year"}
MANIFEST_COLUMNS = [
    "id",
    "plume_id",
    "label",
    "latitude",
    "longitude",
    "datetime",
    "overlap_mode",
    "anchor_sensor",
    "dx_anchor_px",
    "dy_anchor_px",
    "s2_0_path",
    "s2_90_path",
    "s2_360_path",
    "s2_plume_path",
    "l89_0_path",
    "l89_90_path",
    "l89_360_path",
    "l89_plume_path",
    "emit_0_path",
    "emit_90_path",
    "emit_360_path",
    "emit_plume_path",
    "s5p_0_path",
    "s5p_90_path",
    "s5p_360_path",
    "s5p_plume_path",
    "source_group",
    "negative_evidence",
    "selected_s2_id",
    "selected_s2_cloud_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-events", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-source-manifest", type=Path, required=True)
    parser.add_argument("--output-source-audit", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_chw(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    if image.ndim != 3:
        raise ValueError(f"{path}: expected 3-D TIFF, got {image.shape}")
    if image.shape[0] == 12:
        return image
    if image.shape[-1] == 12:
        return image.transpose(2, 0, 1)
    raise ValueError(f"{path}: expected 12 channels, got {image.shape}")


def legacy_t0(full: np.ndarray) -> np.ndarray:
    """Recreate the historical raw-t0 layout from harmonized GEE L2A DN."""
    working = full.astype(np.uint32, copy=False)
    output = np.zeros(full.shape, dtype=np.uint16)
    mapping = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
               (7, 8), (10, 10), (11, 11))
    for output_index, input_index in mapping:
        output[output_index] = np.clip(
            working[input_index] + 1000, 0, np.iinfo(np.uint16).max
        ).astype(np.uint16)
    return output


def atomic_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.tif")
    tifffile.imwrite(temporary, image)
    temporary.replace(path)


def clean_status(value: object) -> str:
    return str(value).strip().lower()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.selected_events, low_memory=False)
    downloads = pd.read_csv(args.download_manifest, low_memory=False)
    latest = downloads.drop_duplicates(["plume_id", "timepoint"], keep="last")
    latest = latest[latest["status"].map(clean_status).isin({"downloaded", "skip_existing_valid"})]
    by_key = {
        (str(row.plume_id), str(row.timepoint)): row
        for row in latest.itertuples(index=False)
    }
    expected = {
        (str(plume_id), timepoint)
        for plume_id in events["plume_id"]
        for timepoint in TIMEPOINTS
    }
    missing = sorted(expected - set(by_key))
    if missing:
        raise ValueError(f"missing successful downloads: {missing[:10]}")
    if args.output_source_manifest.exists() or args.output_source_audit.exists():
        raise FileExistsError("refusing to overwrite source manifest/audit")

    output_rows: list[dict[str, Any]] = []
    event_sources: dict[str, dict[str, Any]] = {}
    for event in events.itertuples(index=False):
        plume_id = str(event.plume_id)
        sources = {tp: by_key[(plume_id, tp)] for tp in TIMEPOINTS}
        t0_input = Path(str(sources["t0"].raw_path)).resolve()
        t0_output = (
            args.output_root.resolve() / "legacy_raw" / "t0" / plume_id / "s2.tif"
        )
        if t0_output.exists():
            raise FileExistsError(t0_output)
        transformed = legacy_t0(read_chw(t0_input))
        zero_channels = [index for index in range(12) if not np.any(transformed[index])]
        if zero_channels != [8, 9]:
            raise ValueError(f"{plume_id}: t0 zero channels={zero_channels}")
        atomic_write(t0_output, transformed)

        seasonal_path = Path(str(sources["seasonal"].raw_path)).resolve()
        year_path = Path(str(sources["year"].raw_path)).resolve()
        for slot, path in (("seasonal", seasonal_path), ("year", year_path)):
            image = read_chw(path)
            if any(not np.any(image[index]) for index in range(12)):
                raise ValueError(f"{plume_id}/{slot}: unexpected all-zero channel")

        event_sources[plume_id] = {
            "matched_ground_truth_sample_id": plume_id,
            "source_group": "random_time_no_known_controlled_release",
            "t0_raw_path": str(t0_output),
            "seasonal_raw_path": str(seasonal_path),
            "year_raw_path": str(year_path),
            "t0_acquisition_time_utc": str(sources["t0"].selected_time_utc),
            "seasonal_acquisition_time_utc": str(sources["seasonal"].selected_time_utc),
            "year_acquisition_time_utc": str(sources["year"].selected_time_utc),
            "t0_selected_id": str(sources["t0"].selected_id),
            "seasonal_selected_id": str(sources["seasonal"].selected_id),
            "year_selected_id": str(sources["year"].selected_id),
            "negative_evidence": str(event.negative_evidence),
        }
        row = {column: "" for column in MANIFEST_COLUMNS}
        row.update(
            {
                "id": f"s2_{plume_id}",
                "plume_id": plume_id,
                "label": 0,
                "latitude": float(event.plume_latitude),
                "longitude": float(event.plume_longitude),
                "datetime": str(sources["t0"].selected_time_utc),
                "overlap_mode": False,
                "anchor_sensor": "s2",
                "dx_anchor_px": 0,
                "dy_anchor_px": 0,
                "source_group": "random_time_no_known_controlled_release",
                "negative_evidence": str(event.negative_evidence),
                "selected_s2_id": str(sources["t0"].selected_id),
                "selected_s2_cloud_pct": sources["t0"].cloud_pct,
            }
        )
        output_rows.append(row)

    output = pd.DataFrame.from_records(output_rows, columns=MANIFEST_COLUMNS)
    args.output_source_manifest.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_source_manifest, index=False)
    audit = {
        "pass": True,
        "semantic_label": "no_known_controlled_release_not_metered_zero",
        "selected_events": str(args.selected_events.resolve()),
        "selected_events_sha256": sha256(args.selected_events),
        "download_manifest": str(args.download_manifest.resolve()),
        "download_manifest_sha256": sha256(args.download_manifest),
        "output_manifest": str(args.output_source_manifest.resolve()),
        "rows": len(output),
        "events": len(event_sources),
        "t0_transform": {
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "dn_offset_added": 1000,
            "layout": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "ZERO", "ZERO", "B11", "B12"],
        },
        "history_layout": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
        "event_sources": event_sources,
    }
    args.output_source_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(output), "manifest": str(args.output_source_manifest), "audit": str(args.output_source_audit)}, indent=2))


if __name__ == "__main__":
    main()
