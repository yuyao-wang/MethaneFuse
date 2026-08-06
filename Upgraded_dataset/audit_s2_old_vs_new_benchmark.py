#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OLD_TRAIN = Path(
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "finalDataset_query/legacy_param_360m/manifest_time_train.csv"
)
OLD_TEST = Path(
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/"
    "finalDataset_query/legacy_param_360m/manifest_time_test.csv"
)
OLD_SOURCE = Path(
    "/home/yuyao/methane_train/preprocess_dataset_s2/"
    "CM_S2_L2A_-7_gee90360_std512.csv"
)
NEW_SOURCE = Path(
    "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
    "s2_6time_legacy_rebuild/s2_6time_new_all6_gee_export_input.csv"
)
NEW_TRAIN = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/s2_gee_legacy_notebook_6time/"
    "temporal_cutoff_split/cutoff_2025-12-22/train.csv"
)
NEW_TEST = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/s2_gee_legacy_notebook_6time/"
    "temporal_cutoff_split/cutoff_2025-12-22/test.csv"
)
OUTPUT = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/diagnostics/"
    "s2_old_vs_new_benchmark_audit.json"
)

METRICS = {
    "t0_only": Path(
        "/diniuvol/yuyao/checkpoints/s2_gee_legacy_notebook_6time/"
        "s2_gee_t0_only_all12_concat_gate3/metrics_history.json"
    ),
    "legacy_3time": Path(
        "/diniuvol/yuyao/checkpoints/s2_gee_legacy_notebook_6time/"
        "s2_gee_legacy3_all12_concat_gate3/metrics_history.json"
    ),
    "six_time": Path(
        "/diniuvol/yuyao/checkpoints/s2_gee_legacy_notebook_6time/"
        "s2_gee_legacy_notebook_6time_all12_concat_gate3/metrics_history.json"
    ),
}

OLD_OVERLAP = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/diagnostics/"
    "s2_old_near90_event_overlap/old_checkpoint_overlap_fp32.json"
)
OLD_NONOVERLAP = Path(
    "/home/yuyao/panopticon/Upgraded_dataset/diagnostics/"
    "s2_old_near90_event_overlap/old_checkpoint_nonoverlap_fp32.json"
)


def event_id(plume_id: Any) -> str:
    return str(plume_id).strip().rsplit("-", 1)[0]


def read_s2_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return frame[frame["s2_0_path"].notna()].copy()


def date_range(frame: pd.DataFrame, column: str) -> dict[str, str]:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return {
        "min": values.min().isoformat(),
        "max": values.max().isoformat(),
    }


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def numeric_summary(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def offset_summary(frame: pd.DataFrame, dx: pd.Series, dy: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in (0, 1):
        selected = frame["label"].astype(int).eq(label)
        radial_pixels = np.sqrt(
            pd.to_numeric(dx[selected], errors="coerce") ** 2
            + pd.to_numeric(dy[selected], errors="coerce") ** 2
        )
        result[str(label)] = {
            "rows": int(selected.sum()),
            "dx_pixels": numeric_summary(dx[selected]),
            "dy_pixels": numeric_summary(dy[selected]),
            "radial_pixels": numeric_summary(radial_pixels),
            "radial_meters": numeric_summary(radial_pixels * 10.0),
        }
    return result


def split_summary(frame: pd.DataFrame, event_column: str, time_column: str) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "plumes": int(frame["plume_id"].nunique()),
        "events": int(frame[event_column].nunique()),
        "labels": label_counts(frame),
        "date_range": date_range(frame, time_column),
    }


def best_epoch(path: Path) -> dict[str, Any]:
    history = json.loads(path.read_text())
    row = max(history, key=lambda item: item["test_best_f1"])
    return {
        "epoch": int(row["epoch"]),
        "train_f1": float(row["train_f1"]),
        "test_fixed_f1": float(row["test_f1"]),
        "test_best_f1": float(row["test_best_f1"]),
        "test_auroc": float(row["test_auroc"]),
    }


def id_set(frame: pd.DataFrame) -> set[str]:
    return set(frame["plume_id"].astype(str).str.strip())


def lower_set(values: set[str]) -> set[str]:
    return {value.lower() for value in values}


def main() -> None:
    old_train = read_s2_rows(OLD_TRAIN)
    old_test = read_s2_rows(OLD_TEST)
    old_all = pd.concat([old_train, old_test], ignore_index=True)
    old_all["event_group_id"] = old_all["plume_id"].map(event_id)

    old_source = pd.read_csv(OLD_SOURCE, low_memory=False)
    old_source["event_group_id"] = old_source["plume_id"].map(event_id)
    new_source = pd.read_csv(NEW_SOURCE, low_memory=False)
    new_train = pd.read_csv(NEW_TRAIN, low_memory=False)
    new_test = pd.read_csv(NEW_TEST, low_memory=False)
    new_all = pd.concat([new_train, new_test], ignore_index=True)

    old_train_events = set(old_train["plume_id"].map(event_id))
    old_test_events = set(old_test["plume_id"].map(event_id))
    new_train_events = set(new_train["event_group_id"].astype(str))
    new_test_events = set(new_test["event_group_id"].astype(str))

    old_trained_ids = id_set(old_all)
    old_source_ids = id_set(old_source)
    new_source_ids = id_set(new_source)
    new_retained_ids = id_set(new_all)
    old_source_events = set(old_source["event_group_id"].astype(str))
    new_source_events = set(new_source["event_group_id"].astype(str))

    old_dx = pd.to_numeric(old_all["dx_anchor_px"], errors="coerce")
    old_dy = pd.to_numeric(old_all["dy_anchor_px"], errors="coerce")
    new_dx = pd.to_numeric(new_all["crop_x"], errors="coerce") + 16 - 256
    new_dy = pd.to_numeric(new_all["crop_y"], errors="coerce") + 16 - 256

    overlap_eval = json.loads(OLD_OVERLAP.read_text())
    nonoverlap_eval = json.loads(OLD_NONOVERLAP.read_text())

    audit = {
        "finding": {
            "main": (
                "The current pure-GEE benchmark is the new_all6_gee complement cohort, "
                "not the old legacy cohort used by the near-0.90 experiment."
            ),
            "not_six_time": (
                "Controlled t0-only, legacy-3-time, and 6-time runs on the same new split "
                "all peak near 0.81 best-threshold F1."
            ),
            "not_event_leakage_main_cause": (
                "The old checkpoint remains near 0.90 F1 on test rows whose canonical "
                "events never occur in old training."
            ),
            "secondary_protocol_differences": [
                "old crop is 360 m (36 native S2 pixels) while current crop is 320 m (32 pixels)",
                "old and current train/test populations and cutoffs differ",
                "current concat_channels fusion discards explicit visit identity",
            ],
        },
        "provenance": {
            "old_crop_launch_evidence": {
                "git_repo": "/home/yuyao/methane_train",
                "git_commit": "b955193",
                "script": "run.sh",
                "output_root": str(OLD_TRAIN.parent),
                "argument": "--query_size_m 360",
                "s2_native_pixels": 36,
                "s2_gsd_m": 10,
                "target_pixels": 224,
            },
            "new_crop_config": {
                "script": (
                    "/home/yuyao/panopticon/Upgraded_dataset/"
                    "s2_gee_legacy_notebook_6time.py"
                ),
                "input_csv": str(NEW_SOURCE),
                "cohort_counts": {
                    str(key): int(value)
                    for key, value in new_source["cohort"].value_counts().items()
                },
                "s2_native_pixels": 32,
                "s2_gsd_m": 10,
                "target_pixels": 224,
            },
            "gee_export_loop": {
                "script": (
                    "/home/yuyao/methane_train/Upgrade_data_pipeline/code/"
                    "S2_preprocess/run_s2_new_gee_export_loop.sh"
                ),
                "input_csv": str(NEW_SOURCE),
                "rows": int(len(new_source)),
            },
        },
        "cohort_comparison": {
            "old_legacy_source_plumes": int(len(old_source_ids)),
            "old_near90_trained_plumes": int(len(old_trained_ids)),
            "new_gee_source_plumes": int(len(new_source_ids)),
            "new_gee_retained_plumes": int(len(new_retained_ids)),
            "exact_plume_overlap_old_source_vs_new_source": int(
                len(old_source_ids & new_source_ids)
            ),
            "casefold_plume_overlap_old_source_vs_new_source": int(
                len(lower_set(old_source_ids) & lower_set(new_source_ids))
            ),
            "exact_plume_overlap_old_trained_vs_new_retained": int(
                len(old_trained_ids & new_retained_ids)
            ),
            "canonical_event_overlap_old_source_vs_new_source": int(
                len(old_source_events & new_source_events)
            ),
            "canonical_event_overlap_examples": sorted(
                old_source_events & new_source_events
            )[:20],
            "old_source_date_range": date_range(old_source, "datetime"),
            "new_source_date_range": date_range(new_source, "event_time"),
        },
        "old_near90_benchmark": {
            "train": split_summary(old_train.assign(event_group_id=old_train["plume_id"].map(event_id)), "event_group_id", "datetime"),
            "test": split_summary(old_test.assign(event_group_id=old_test["plume_id"].map(event_id)), "event_group_id", "datetime"),
            "train_test_event_overlap": int(len(old_train_events & old_test_events)),
            "test_rows_from_overlapping_events": int(overlap_eval["rows"]),
            "test_rows_from_nonoverlapping_events": int(nonoverlap_eval["rows"]),
            "overlap_event_checkpoint_fp32": overlap_eval["metrics"],
            "nonoverlap_event_checkpoint_fp32": nonoverlap_eval["metrics"],
            "crop_offset_summary": offset_summary(old_all, old_dx, old_dy),
        },
        "current_pure_gee_benchmark": {
            "train": split_summary(new_train, "event_group_id", "event_time"),
            "test": split_summary(new_test, "event_group_id", "event_time"),
            "train_test_event_overlap": int(len(new_train_events & new_test_events)),
            "crop_offset_summary": offset_summary(new_all, new_dx, new_dy),
        },
        "same_split_temporal_ablation": {
            name: best_epoch(path) for name, path in METRICS.items()
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
