#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from s2_point_center_recut_diagnostic import plume_shift_pixels


PRODUCT_COLUMNS = {
    "t0": "t0_product_name",
    "prev1": "prev1_product_name",
    "prev2": "prev2_product_name",
    "prev3": "prev3_product_name",
    "seasonal": "seasonal_product_name",
    "year": "year_product_name",
}


def old_plume_ids(master_csv: str, old_train_csv: str, old_test_csv: str) -> set[str]:
    master = pd.read_csv(
        master_csv,
        usecols=["plume_id", "datetime", "plume_latitude", "plume_longitude"],
        low_memory=False,
    )
    master["time"] = pd.to_datetime(master["datetime"], utc=True, errors="coerce").dt.round("s")
    master["lat6"] = master["plume_latitude"].round(6)
    master["lon6"] = master["plume_longitude"].round(6)
    identifiers: set[str] = set()
    for csv_path in (old_train_csv, old_test_csv):
        frame = pd.read_csv(
            csv_path,
            usecols=["datetime", "latitude", "longitude"],
            low_memory=False,
        )
        frame["time"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.round("s")
        frame["lat6"] = frame["latitude"].round(6)
        frame["lon6"] = frame["longitude"].round(6)
        mapped = frame.merge(master, on=["time", "lat6", "lon6"], how="left")
        identifiers.update(mapped["plume_id"].dropna().astype(str))
    return identifiers


def split_audit(
    name: str,
    patch_csv: str,
    shifts: dict[str, dict[str, tuple[int, int, float]]],
) -> dict:
    frame = pd.read_csv(
        patch_csv,
        usecols=["plume_id", "label", "crop_x", "crop_y"],
        low_memory=False,
    )
    frame["shift"] = frame["plume_id"].astype(str).map(
        {plume_id: values["t0"] for plume_id, values in shifts.items()}
    )
    frame = frame[frame["shift"].notna()].copy()
    frame["shift_x"] = frame["shift"].map(lambda value: value[0])
    frame["shift_y"] = frame["shift"].map(lambda value: value[1])
    frame["offset_km"] = frame["shift"].map(lambda value: value[2])
    frame["actual_x"] = 256 + frame["shift_x"]
    frame["actual_y"] = 256 + frame["shift_y"]
    frame["actual_point_inside"] = (
        frame["crop_x"].le(frame["actual_x"])
        & frame["actual_x"].lt(frame["crop_x"] + 32)
        & frame["crop_y"].le(frame["actual_y"])
        & frame["actual_y"].lt(frame["crop_y"] + 32)
    )
    positive = frame[frame["label"].eq(1)]
    negative = frame[frame["label"].eq(0)]
    plume_ids = set(frame["plume_id"].astype(str))
    plume_offsets = pd.Series(
        {
            plume_id: shifts[plume_id]["t0"][2]
            for plume_id in plume_ids
            if plume_id in shifts
        }
    )
    recoverable = sum(
        all(
            -234 <= shift_x <= 234 and -234 <= shift_y <= 234
            for shift_x, shift_y, _ in shifts[plume_id].values()
        )
        for plume_id in plume_ids
        if plume_id in shifts
    )
    return {
        "name": name,
        "rows": len(frame),
        "plumes": len(plume_ids),
        "positive_rows": len(positive),
        "positive_actual_point_inside": int(positive["actual_point_inside"].sum()),
        "positive_spatially_mismatched": int((~positive["actual_point_inside"]).sum()),
        "positive_spatial_mismatch_rate": float((~positive["actual_point_inside"]).mean()),
        "negative_actual_point_inside": int(negative["actual_point_inside"].sum()),
        "negative_actual_point_inside_rate": float(negative["actual_point_inside"].mean()),
        "plume_offset_km": {
            str(key): float(value)
            for key, value in plume_offsets.describe(
                percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            ).to_dict().items()
        },
        "plumes_offset_over_0_32km": int((plume_offsets > 0.32).sum()),
        "plumes_offset_over_1km": int((plume_offsets > 1.0).sum()),
        "all6_center_patch_recoverable": recoverable,
        "all6_center_patch_recoverable_rate": recoverable / max(1, len(plume_ids)),
    }


def main(args: argparse.Namespace) -> None:
    source = pd.read_csv(args.source_csv, low_memory=False)
    shifts: dict[str, dict[str, tuple[int, int, float]]] = {}
    failures = []
    for row in source.to_dict("records"):
        plume_id = str(row["plume_id"])
        try:
            shifts[plume_id] = {
                name: plume_shift_pixels(row, product_column)
                for name, product_column in PRODUCT_COLUMNS.items()
            }
        except Exception as error:
            failures.append({"plume_id": plume_id, "reason": f"{type(error).__name__}:{error}"})
    historical_ids = old_plume_ids(
        args.master_csv,
        args.old_train_csv,
        args.old_test_csv,
    )
    current_ids = set(source["plume_id"].astype(str))
    output = {
        "source_rows": len(source),
        "source_unique_plumes": int(source["plume_id"].nunique()),
        "shift_failures": failures,
        "historical_unique_plumes": len(historical_ids),
        "historical_current_overlap": len(historical_ids & current_ids),
        "train": split_audit("train", args.train_csv, shifts),
        "test": split_audit("test", args.test_csv, shifts),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--old-train-csv", required=True)
    parser.add_argument("--old-test-csv", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
