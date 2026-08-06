#!/usr/bin/env python3
"""Build fine-grained paired S2 band-group and temporal-frame ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_s2_mechanism_probe import (
    FRAMES,
    T0_FORCED_ZERO,
    atomic_tiff,
    force_contract,
    json_dump,
    read_chw,
    sha256,
    spatial_constant,
)


VISIBLE = tuple(range(0, 4))
RED_EDGE_NIR = tuple(range(4, 10))
B11 = (10,)
B12 = (11,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = args.input_manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    source = pd.read_csv(input_manifest, low_memory=False)

    modes = (
        "original",
        "constant_t0_all",
        "constant_t90_all",
        "constant_t360_all",
        "constant_history_all",
        "constant_visible_alltime",
        "constant_rededge_nir_alltime",
        "constant_b11_alltime",
        "constant_b12_alltime",
        "constant_t0_visible",
        "constant_t0_rededge_nir",
        "constant_t0_b11",
        "constant_t0_b12",
        "swap_b11_b12_alltime",
        "swap_t0_b11_b12",
    )
    output_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for row in source.to_dict(orient="records"):
        parent_id = str(row.get("parent_id") or row["id"]).split("__scale_1p00")[0]
        arrays = {frame: read_chw(str(row[column])) for frame, column in FRAMES}
        for mode in modes:
            changed_frames: set[str] = set()
            result = {frame: array.copy() for frame, array in arrays.items()}
            if mode == "original":
                pass
            elif mode.startswith("constant_"):
                if mode == "constant_t0_all":
                    frame_names, channels = ("t0",), tuple(range(12))
                elif mode == "constant_t90_all":
                    frame_names, channels = ("t90",), tuple(range(12))
                elif mode == "constant_t360_all":
                    frame_names, channels = ("t360",), tuple(range(12))
                elif mode == "constant_history_all":
                    frame_names, channels = ("t90", "t360"), tuple(range(12))
                elif mode == "constant_visible_alltime":
                    frame_names, channels = ("t0", "t90", "t360"), VISIBLE
                elif mode == "constant_rededge_nir_alltime":
                    frame_names, channels = ("t0", "t90", "t360"), RED_EDGE_NIR
                elif mode == "constant_b11_alltime":
                    frame_names, channels = ("t0", "t90", "t360"), B11
                elif mode == "constant_b12_alltime":
                    frame_names, channels = ("t0", "t90", "t360"), B12
                elif mode == "constant_t0_visible":
                    frame_names, channels = ("t0",), VISIBLE
                elif mode == "constant_t0_rededge_nir":
                    frame_names, channels = ("t0",), RED_EDGE_NIR
                elif mode == "constant_t0_b11":
                    frame_names, channels = ("t0",), B11
                elif mode == "constant_t0_b12":
                    frame_names, channels = ("t0",), B12
                else:
                    raise AssertionError(mode)
                for frame in frame_names:
                    result[frame] = spatial_constant(result[frame], channels)
                    changed_frames.add(frame)
            elif mode == "swap_b11_b12_alltime":
                for frame in ("t0", "t90", "t360"):
                    result[frame][[10, 11]] = result[frame][[11, 10]]
                    changed_frames.add(frame)
            elif mode == "swap_t0_b11_b12":
                result["t0"][[10, 11]] = result["t0"][[11, 10]]
                changed_frames.add("t0")
            else:
                raise AssertionError(mode)

            paths: dict[str, str] = {}
            frame_diagnostics: dict[str, object] = {}
            for frame, column in FRAMES:
                transformed = force_contract(result[frame], frame)
                if frame not in changed_frames:
                    paths[column] = str(Path(str(row[column])).resolve())
                else:
                    path = output_root / "crops" / mode / parent_id / f"{frame}.tif"
                    atomic_tiff(path, transformed)
                    paths[column] = str(path)
                frame_diagnostics[frame] = {
                    "changed": frame in changed_frames,
                    "mean_delta": float(transformed.mean(dtype=np.float64) - arrays[frame].mean(dtype=np.float64)),
                    "std_delta": float(transformed.std(dtype=np.float64) - arrays[frame].std(dtype=np.float64)),
                    "t0_contract_ok": bool(frame != "t0" or np.all(transformed[list(T0_FORCED_ZERO)] == 0)),
                }

            out = dict(row)
            out.update(paths)
            out["id"] = f"{parent_id}__{mode}"
            out["parent_id"] = parent_id
            out["probe_mode"] = mode
            out["counterfactual_probe_only"] = mode != "original"
            output_rows.append(out)
            diagnostics.append(
                {"parent_id": parent_id, "label": int(row["label"]), "mode": mode, "frames": frame_diagnostics}
            )

    combined = pd.DataFrame(output_rows)
    combined.to_csv(output_root / "manifest_all_modes.csv", index=False)
    for mode, group in combined.groupby("probe_mode", sort=False):
        group.to_csv(output_root / f"manifest_{mode}.csv", index=False)
    json_dump(
        output_root / "dataset_audit.json",
        {
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": sha256(input_manifest),
            "rows": len(combined),
            "source_rows": len(source),
            "modes": list(modes),
            "band_groups": {
                "visible": {"indices": list(VISIBLE), "bands": ["B01", "B02", "B03", "B04"]},
                "rededge_nir": {
                    "indices": list(RED_EDGE_NIR),
                    "bands": ["B05", "B06", "B07", "B08", "B8A", "B09"],
                },
                "B11": {"indices": [10]},
                "B12": {"indices": [11]},
            },
            "t0_forced_zero_channels": list(T0_FORCED_ZERO),
            "warning": "Counterfactual band/time probe only; transformed rows are not physical observations.",
            "diagnostics": diagnostics,
        },
    )
    print(json.dumps({"rows": len(combined), "modes": len(modes), "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
