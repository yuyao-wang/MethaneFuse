import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset.crop_s2_legacy360_matched_gee import (
    alignment_shift,
    read_bhw,
)


RETAINED_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 10, 11]


def legacy_t0_contract(image: np.ndarray) -> np.ndarray:
    source = image.astype(np.float32, copy=False)
    transformed = np.zeros_like(source)
    retained = source[RETAINED_CHANNELS]
    transformed[RETAINED_CHANNELS] = np.where(
        np.abs(retained) < 0.5,
        0,
        retained + 1000,
    )
    transformed[7] = np.where(
        np.abs(source[8]) < 0.5,
        0,
        source[8] + 1000,
    )
    return transformed


def audit_one(item: tuple[str, str, str, int, int], image_size: int, search_radius: int):
    plume_id, raw_path, reference_path, seasonal_y, seasonal_x = item
    raw = legacy_t0_contract(read_bhw(Path(raw_path)))
    shift_y, shift_x, mae, correlation = alignment_shift(
        raw,
        Path(reference_path),
        image_size=image_size,
        search_radius=search_radius,
    )
    return {
        "plume_id": plume_id,
        "t0_shift_y": shift_y,
        "t0_shift_x": shift_x,
        "seasonal_shift_y": seasonal_y,
        "seasonal_shift_x": seasonal_x,
        "delta_y": shift_y - seasonal_y,
        "delta_x": shift_x - seasonal_x,
        "mae_sampled": mae,
        "correlation_sampled": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-input-csv", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--crop-diagnostics", required=True)
    parser.add_argument("--legacy-t0-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--search-radius", type=int, default=3)
    args = parser.parse_args()

    alignments = {}
    with Path(args.crop_diagnostics).open() as handle:
        for line in handle:
            record = json.loads(line)
            alignment = record.get("alignment") or {}
            if record.get("status") == "ok" and alignment:
                alignments[str(record["plume_id"])] = (
                    int(alignment["shift_y"]),
                    int(alignment["shift_x"]),
                )

    frame = pd.read_csv(args.export_input_csv, usecols=["plume_id"])
    candidates = []
    raw_root = Path(args.raw_root)
    reference_root = Path(args.legacy_t0_root)
    for plume_id in frame["plume_id"].astype(str):
        if plume_id not in alignments:
            continue
        raw_path = raw_root / "S2_GEE_6time" / "t0" / plume_id / "s2_0.tif"
        reference_path = reference_root / plume_id / "s2.tif"
        if not raw_path.is_file() or not reference_path.is_file():
            continue
        seasonal_y, seasonal_x = alignments[plume_id]
        candidates.append(
            (
                plume_id,
                str(raw_path),
                str(reference_path),
                seasonal_y,
                seasonal_x,
            )
        )
    sampled = (
        pd.Series(range(len(candidates)))
        .sample(n=min(args.samples, len(candidates)), random_state=args.seed)
        .tolist()
    )
    items = [candidates[index] for index in sampled]

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                audit_one,
                item,
                args.image_size,
                args.search_radius,
            )
            for item in items
        ]
        for index, future in enumerate(futures, start=1):
            records.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"[Alignment] processed {index}/{len(futures)}", flush=True)

    delta_counts = Counter((record["delta_y"], record["delta_x"]) for record in records)
    correlations = np.array(
        [record["correlation_sampled"] for record in records],
        dtype=np.float64,
    )
    maes = np.array([record["mae_sampled"] for record in records], dtype=np.float64)
    report = {
        "candidate_plumes": len(candidates),
        "sampled_plumes": len(records),
        "same_shift_fraction": float(
            np.mean(
                [
                    record["delta_y"] == 0 and record["delta_x"] == 0
                    for record in records
                ]
            )
        ),
        "delta_counts": {
            f"{delta_y},{delta_x}": count
            for (delta_y, delta_x), count in delta_counts.most_common()
        },
        "correlation": {
            "mean": float(correlations.mean()),
            "median": float(np.median(correlations)),
            "min": float(correlations.min()),
        },
        "mae": {
            "mean": float(maes.mean()),
            "median": float(np.median(maes)),
            "max": float(maes.max()),
        },
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
