#!/usr/bin/env python3
"""CPU-only tests for the sidecar inner-train capability split."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from research.pretraining_20260727 import l89_ragged_cls_experiment as base
from research.tempo_20260728 import prepare_l89_sidecar_inner_split as target


def synthetic_payload(frame: pd.DataFrame, csv_path: Path) -> dict:
    rows = len(frame)
    timepoints = 6
    features = torch.arange(rows * timepoints * 4, dtype=torch.float32).reshape(
        rows, timepoints, 4
    )
    labels = torch.tensor(frame["label"].tolist(), dtype=torch.long)
    valid = torch.ones(rows, timepoints, dtype=torch.bool)
    zeros = torch.zeros(rows, timepoints, dtype=torch.bool)
    timestamps = torch.arange(rows * timepoints, dtype=torch.long).reshape(
        rows, timepoints
    )
    path_columns = list(target.cache.parse_columns(
        "path_t0,path_prev1,path_prev2,path_prev3,path_seasonal,path_year"
    ))
    time_columns = list(target.cache.parse_columns(
        "t0_image_time,prev1_image_time,prev2_image_time,prev3_image_time,"
        "seasonal_image_time,year_image_time"
    ))
    contract = {
        "script_version": "l89-ragged-cls-v1",
        "csv_sha256": base.sha256_file(csv_path),
        "weights_sha256": "weights-sha",
        "input_table_sha256": "table-sha",
        "path_columns": path_columns,
        "time_columns": time_columns,
        "role_names": ["t0", "prev1", "prev2", "prev3", "seasonal", "year"],
        "band_indices": list(range(7)),
        "channel_ids": [1.0] * 7,
        "normalization_mean": [0.0] * 7,
        "normalization_std": [1.0] * 7,
        "normalization_source": "unit",
        "image_size": 224,
        "min_valid_fraction": 0.75,
        "validity_band_index": 0,
        "zero_invalid_pixels": True,
        "local_cache_bypass_root": "/diniuvol/yuyao",
        "duplicate_rule": "unit",
        "source_rows": rows,
        "selected_rows": rows,
        "row_selection": "all",
        "row_selection_seed": 0,
    }
    return {
        "format_version": 1,
        "script_version": "l89-ragged-cls-v1",
        "split": "train",
        "features": features,
        "labels": labels,
        "ids": frame["id"].astype(str).tolist(),
        "plume_ids": frame["plume_id"].astype(str).tolist(),
        "event_ids": frame["event_group_id"].astype(str).tolist(),
        "event_id_rule": "declared",
        "timestamps_utc_ns": timestamps,
        "timestamps_utc_iso": [["x"] * timepoints for _ in range(rows)],
        "timestamp_valid_mask": valid.clone(),
        "delta_days": torch.zeros(rows, timepoints),
        "role_names": contract["role_names"],
        "role_index": torch.arange(timepoints),
        "t0_index": 0,
        "image_valid_mask": valid.clone(),
        "valid_mask": valid.clone(),
        "duplicate_mask": zeros.clone(),
        "duplicate_group_mask": zeros.clone(),
        "unique_mask": valid.clone(),
        "valid_fraction": torch.ones(rows, timepoints),
        "load_status": torch.ones(rows, timepoints, dtype=torch.int8),
        "path_columns": path_columns,
        "time_columns": time_columns,
        "input_contract": contract,
        "input_contract_sha256": base.sha256_bytes(
            base.canonical_json_bytes(contract)
        ),
        "csv_path": str(csv_path),
        "csv_sha256": base.sha256_file(csv_path),
        "weights_path": "/weights.pth",
        "weights_sha256": "weights-sha",
        "input_table_sha256": "table-sha",
        "feature_sha256": base.tensor_sha256(features),
        "label_sha256": base.tensor_sha256(labels),
        "timestamp_sha256": base.tensor_sha256(timestamps),
    }


class SidecarSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sidecar_split_unit_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_source(self) -> tuple[Path, Path]:
        rows = []
        row_id = 0
        for event_index in range(10):
            date = f"{2020 + event_index}-01-01"
            for within in range(600):
                row_id += 1
                row = {
                    "id": row_id,
                    "label": (event_index + within) % 2,
                    "plume_id": f"event{event_index}-{within}",
                    "event_group_id": f"event{event_index}",
                    "event_time": date,
                }
                for column in (
                    "path_t0",
                    "path_prev1",
                    "path_prev2",
                    "path_prev3",
                    "path_seasonal",
                    "path_year",
                ):
                    row[column] = f"/local/{row_id}/{column}.tif"
                for column in (
                    "t0_image_time",
                    "prev1_image_time",
                    "prev2_image_time",
                    "prev3_image_time",
                    "seasonal_image_time",
                    "year_image_time",
                ):
                    row[column] = date
                rows.append(row)
        frame = pd.DataFrame(rows)
        csv_path = self.root / "train.csv"
        frame.to_csv(csv_path, index=False)
        payload = synthetic_payload(frame, csv_path)
        cache_path = self.root / "train.pt"
        torch.save(payload, cache_path)
        return csv_path, cache_path

    def test_whole_event_capability_and_aligned_caches(self) -> None:
        csv_path, cache_path = self.make_source()
        output = self.root / "sidecar_split"
        audit = target.run(
            target.build_parser().parse_args(
                [
                    "--train-csv",
                    str(csv_path),
                    "--train-cache",
                    str(cache_path),
                    "--output-root",
                    str(output),
                    "--minimum-capability-rows",
                    "512",
                ]
            )
        )
        self.assertEqual(audit["event_overlap"], 0)
        self.assertFalse(
            audit[
                "formal_inner_development_used_for_sidecar_checkpoint_selection"
            ]
        )
        capability = pd.read_csv(output / "capability.csv")
        fit = pd.read_csv(output / "train.csv")
        self.assertEqual(set(capability["event_group_id"]), {"event8", "event9"})
        self.assertFalse(
            set(capability["event_group_id"]) & set(fit["event_group_id"])
        )
        train_cache = base.torch_load_trusted(output / "train.pt")
        capability_cache = base.torch_load_trusted(output / "capability.pt")
        base.validate_cache_payload(
            train_cache, path=output / "train.pt", expected_split="train"
        )
        base.validate_cache_payload(
            capability_cache,
            path=output / "capability.pt",
            expected_split="val",
        )
        self.assertEqual(
            train_cache["ids"], fit["id"].astype(str).tolist()
        )
        self.assertEqual(
            capability_cache["ids"], capability["id"].astype(str).tolist()
        )
        self.assertEqual(
            audit["selection"]["locked_train_plan_lengths_by_epoch"],
            [2048, 2048],
        )
        self.assertEqual(
            audit["selection"]["locked_capability_plan_length"], 512
        )
        self.assertTrue(
            audit["source"]["ordered_id_plume_event_label_exact"]
        )
        for key in (
            "labels_sha256",
            "timestamps_utc_ns_sha256",
            "valid_mask_sha256",
            "unique_mask_sha256",
            "delta_days_sha256",
            "valid_fraction_sha256",
        ):
            self.assertIn(
                key,
                audit["cache_summaries"]["train"]["row_tensor_digests"],
            )

    def test_rejects_plume_order_and_stale_declared_digests(self) -> None:
        csv_path, cache_path = self.make_source()
        frame = pd.read_csv(csv_path)
        payload = base.torch_load_trusted(cache_path)

        wrong_plume = dict(payload)
        wrong_plume["plume_ids"] = list(payload["plume_ids"])
        wrong_plume["plume_ids"][0] = "wrong-plume"
        with self.assertRaisesRegex(ValueError, "plume IDs"):
            target.validate_source_alignment(frame, wrong_plume, csv_path)

        stale_label = dict(payload)
        stale_label["label_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "label SHA"):
            target.validate_source_alignment(frame, stale_label, csv_path)

        stale_timestamp = dict(payload)
        stale_timestamp["timestamp_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "timestamp SHA"):
            target.validate_source_alignment(frame, stale_timestamp, csv_path)

    def test_held_out_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "held-out"):
            target.safe_path(self.root / "outer_panel.csv", purpose="unit")


if __name__ == "__main__":
    unittest.main()
