#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrow-csv", required=True)
    parser.add_argument("--wide-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stable-delta-px", type=float, default=1.0)
    args = parser.parse_args()

    narrow = pd.read_csv(args.narrow_csv, low_memory=False)
    wide = pd.read_csv(args.wide_csv, low_memory=False)
    joined = narrow.merge(
        wide,
        on=["id", "plume_id"],
        suffixes=("_narrow", "_wide"),
        validate="one_to_one",
    )
    joined["window_shift_delta_px"] = np.hypot(
        joined["subpixel_shift_x_to_apply_new_px_narrow"]
        - joined["subpixel_shift_x_to_apply_new_px_wide"],
        joined["subpixel_shift_y_to_apply_new_px_narrow"]
        - joined["subpixel_shift_y_to_apply_new_px_wide"],
    )
    joined["registration_stable"] = (
        joined["window_shift_delta_px"] <= args.stable_delta_px
    )
    stable = joined.loc[joined["registration_stable"]]
    unstable = joined.loc[~joined["registration_stable"]]
    summary = {
        "narrow_csv": args.narrow_csv,
        "wide_csv": args.wide_csv,
        "plumes": len(joined),
        "stability_definition": (
            "Euclidean difference between narrow- and wide-window subpixel shifts "
            f"<= {args.stable_delta_px} px"
        ),
        "stable_plumes": len(stable),
        "unstable_plumes": len(unstable),
        "stable_fraction": float(joined["registration_stable"].mean()),
        "window_shift_delta_px": describe(joined["window_shift_delta_px"]),
        "stable_narrow_shift_magnitude_px": describe(
            stable["subpixel_shift_magnitude_px_narrow"]
        ),
        "stable_narrow_shift_magnitude_m": describe(
            stable["subpixel_shift_magnitude_m_narrow"]
        ),
        "stable_improvement_fraction": {
            "mae_dn": float(
                stable["mae_dn_ratio_after_before_narrow"].lt(1.0).mean()
            ),
            "rmse_robust_z": float(
                stable["rmse_robust_z_ratio_after_before_narrow"].lt(1.0).mean()
            ),
            "median_band_correlation": float(
                (
                    stable["after_median_band_correlation_narrow"]
                    > stable["before_median_band_correlation_narrow"]
                ).mean()
            ),
        },
        "search_peak_correlation": {
            "stable": describe(stable["search_peak_correlation_narrow"]),
            "unstable": describe(unstable["search_peak_correlation_narrow"]),
        },
        "repair_input_recommendation": (
            "Use narrow-window shifts only where registration_stable=true; "
            "route unstable plumes to geotransform-based or manual review."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "per_plume_window_stability.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
