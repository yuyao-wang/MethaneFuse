#!/usr/bin/env python3
"""Build all 16 t0 visible-band texture-removal subsets for paired attribution."""

from __future__ import annotations

import argparse
import itertools
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


VISIBLE_BANDS = ((0, "B01"), (1, "B02"), (2, "B03"), (3, "B04"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def mode_name(names: tuple[str, ...]) -> str:
    return "constant_t0_none" if not names else "constant_t0_" + "_".join(names)


def main() -> None:
    args = parse_args()
    input_manifest = args.input_manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    source = pd.read_csv(input_manifest, low_memory=False)

    subsets: list[tuple[tuple[int, ...], tuple[str, ...], int]] = []
    for size in range(len(VISIBLE_BANDS) + 1):
        for chosen in itertools.combinations(VISIBLE_BANDS, size):
            indices = tuple(item[0] for item in chosen)
            names = tuple(item[1] for item in chosen)
            mask = sum(1 << index for index in indices)
            subsets.append((indices, names, mask))

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for row in source.to_dict(orient="records"):
        parent_id = str(row.get("parent_id") or row["id"]).split("__scale_1p00")[0]
        arrays = {frame: read_chw(str(row[column])) for frame, column in FRAMES}
        for indices, names, subset_mask in subsets:
            mode = mode_name(names)
            paths = {column: str(Path(str(row[column])).resolve()) for _, column in FRAMES}
            transformed = arrays["t0"]
            if indices:
                transformed = force_contract(spatial_constant(arrays["t0"], indices), "t0")
                output_path = output_root / "crops" / mode / parent_id / "t0.tif"
                atomic_tiff(output_path, transformed)
                paths["s2_0_path"] = str(output_path)
            out = dict(row)
            out.update(paths)
            out["id"] = f"{parent_id}__{mode}"
            out["parent_id"] = parent_id
            out["probe_mode"] = mode
            out["removed_visible_mask"] = subset_mask
            out["removed_visible_bands"] = "+".join(names) if names else "none"
            out["removed_visible_count"] = len(indices)
            out["counterfactual_probe_only"] = bool(indices)
            rows.append(out)
            diagnostics.append(
                {
                    "parent_id": parent_id,
                    "label": int(row["label"]),
                    "probe_mode": mode,
                    "removed_visible_mask": subset_mask,
                    "removed_visible_bands": list(names),
                    "t0_mean_delta": float(
                        transformed.mean(dtype=np.float64) - arrays["t0"].mean(dtype=np.float64)
                    ),
                    "t0_zero_contract_ok": bool(np.all(transformed[list(T0_FORCED_ZERO)] == 0)),
                }
            )

    combined = pd.DataFrame(rows)
    combined.to_csv(output_root / "manifest_all_subsets.csv", index=False)
    for mode, group in combined.groupby("probe_mode", sort=False):
        group.to_csv(output_root / f"manifest_{mode}.csv", index=False)
    json_dump(
        output_root / "dataset_audit.json",
        {
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": sha256(input_manifest),
            "rows": len(combined),
            "source_rows": len(source),
            "subsets": len(subsets),
            "visible_bands": {name: index for index, name in VISIBLE_BANDS},
            "intervention": "replace each selected t0 band by its own spatial mean",
            "t0_forced_zero_channels": list(T0_FORCED_ZERO),
            "warning": "Counterfactual attribution probe only; transformed rows are not physical observations.",
            "diagnostics": diagnostics,
        },
    )
    print(json.dumps({"rows": len(combined), "subsets": len(subsets), "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
