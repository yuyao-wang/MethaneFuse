#!/usr/bin/env python
"""Run MethaneFuse 480 m query-level inference/evaluation on a MethaneUnion CSV."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.evaluate_classification import build_parser, main as evaluate_classification

DEFAULT_EVAL_CSV = REPO_ROOT / "data" / "MethaneUnion" / "datasets" / "temporal_split" / "480m_GSD" / "test.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "stage2_classification_480m.pt"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "panopticon_vitb14_teacher.pth"
DEFAULT_WV3_SRF = REPO_ROOT / "data" / "manifests" / "WV3_VNIR_SWIR_response.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "examples" / "inference_480m.json"

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
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", prefix="methaneunion_480m_", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    df.to_csv(tmp_path, index=False)
    return tmp_path


def parse_args() -> argparse.Namespace:
    parser = build_parser(
        {
            "eval_csv": str(DEFAULT_EVAL_CSV),
            "checkpoint": str(DEFAULT_CHECKPOINT),
            "weights": str(DEFAULT_WEIGHTS),
            "stage": "b",
            "batch_size": 16,
            "num_workers": 4,
            "row_fusion_mode": "max",
            "output_json": str(DEFAULT_OUTPUT),
            "wv3_srf_csv": str(DEFAULT_WV3_SRF),
        }
    )
    parser.description = __doc__
    parser.add_argument(
        "--no_normalize_columns",
        action="store_true",
        help="Disable MethaneUnion release-column normalization before inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_csv = Path(args.eval_csv)
    tmp_csv = None if args.no_normalize_columns else normalize_manifest_columns(eval_csv)
    if tmp_csv is not None:
        args.eval_csv = str(tmp_csv)
    evaluate_classification(args)


if __name__ == "__main__":
    main()
