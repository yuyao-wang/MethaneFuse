#!/usr/bin/env python
"""Load one MethaneUnion row with the MethaneFuse dataset loader and print tensor shapes."""

from __future__ import annotations

import argparse
import sys
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.finetune_loramoe_adapter import (
    DEFAULT_WV3_BANDS,
    TriSensorTemporalCsvDataset,
    load_wv3_channel_ids_from_srf,
)


DEFAULT_CSV = REPO_ROOT / "data" / "MethaneUnion" / "datasets" / "temporal_split" / "480m_GSD" / "test.csv"
DEFAULT_WV3_SRF = REPO_ROOT / "data" / "manifests" / "WV3_VNIR_SWIR_response.csv"

COLUMN_RENAMES = {
    "S2_t0_path": "s2_0_path",
    "S2_pre_path": "s2_90_path",
    "S2_pre_pre_path": "s2_360_path",
    "S2_plume_label_path": "s2_plume_path",
    "L89_t0_path": "l89_0_path",
    "L89_pre_path": "l89_90_path",
    "L89_pre_pre_path": "l89_360_path",
    "L89_plume_label_path": "l89_plume_path",
    "EMIT_t0_path": "emit_0_path",
    "EMIT_pre_path": "emit_90_path",
    "EMIT_pre_pre_path": "emit_360_path",
    "EMIT_plume_label_path": "emit_plume_path",
    "S5p_temporal_path": "s5p_0_path",
}


def normalize_manifest_columns(csv_path: Path) -> Path | None:
    header = pd.read_csv(csv_path, nrows=0)
    if not any(col in header.columns for col in COLUMN_RENAMES):
        return None
    df = pd.read_csv(csv_path)
    df = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns})
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", prefix="methaneunion_sample_", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    df.to_csv(tmp_path, index=False)
    return tmp_path


def tensor_shape(value: Any) -> list[int] | str:
    return list(value.shape) if hasattr(value, "shape") else type(value).__name__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="MethaneUnion manifest CSV.")
    parser.add_argument("--index", type=int, default=0, help="Row index to load.")
    parser.add_argument("--wv3_srf_csv", default=str(DEFAULT_WV3_SRF))
    parser.add_argument("--wv3_bands", default=",".join(DEFAULT_WV3_BANDS))
    parser.add_argument("--no_normalize_columns", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    tmp_csv = None if args.no_normalize_columns else normalize_manifest_columns(csv_path)
    dataset_csv = tmp_csv or csv_path

    wv3_bands = [x.strip() for x in args.wv3_bands.split(",") if x.strip()]
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_bands).unsqueeze(-1)
    dataset = TriSensorTemporalCsvDataset(
        csv_path=str(dataset_csv),
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )

    sensor_samples, label = dataset[args.index]
    summary = {
        "csv": str(csv_path),
        "index": args.index,
        "label": int(label),
        "sensors": [],
    }
    for sensor_name, frames in sensor_samples:
        summary["sensors"].append(
            {
                "sensor": sensor_name,
                "frames": len(frames),
                "imgs": [tensor_shape(frame["imgs"]) for frame in frames],
                "chn_ids": [tensor_shape(frame["chn_ids"]) for frame in frames],
            }
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
