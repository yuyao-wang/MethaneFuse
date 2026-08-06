#!/usr/bin/env python3
"""CPU contract tests for the strict two-axis fit/sealed-test runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import query360_two_axis_full_legacy as runner
from query360_data import (
    CLASSIFICATION_PATH_COLUMNS,
    Query360Dataset,
    UnsafePathError,
)
from src.data.multisensor import (
    TriSensorTemporalCsvDataset,
    custom_collate_fn,
)


def _cache(path: Path, *, split: str, offset: int) -> None:
    rows, sensors, roles, dim = 8, 4, 3, 8
    generator = torch.Generator().manual_seed(1000 + offset)
    features = torch.randn(
        rows, sensors, roles, dim, generator=generator
    ).to(torch.float16)
    valid = torch.ones(rows, sensors, roles, dtype=torch.bool)
    labels = torch.tensor([0, 1] * 4, dtype=torch.long)
    fused = torch.linspace(-1.0, 1.0, rows)
    sensor_logits = torch.zeros(rows, sensors)
    payload = {
        "schema_version": "query360-two-axis-feature-cache-v1",
        "script_version": runner.SCRIPT_VERSION,
        "split": split,
        "features": features,
        "features_hybrid": features.clone(),
        "features_universal": features.clone(),
        "valid_mask": valid,
        "base_sensor_logits": sensor_logits,
        "base_sensor_valid": torch.ones(rows, sensors, dtype=torch.bool),
        "base_sensor_logits_hybrid": sensor_logits.clone(),
        "base_sensor_logits_universal": sensor_logits.clone(),
        "base_universal_logits": fused.clone(),
        "base_hybrid_logits": fused.clone(),
        "base_fused_logits": fused.clone(),
        "labels": labels,
        "ids": [f"id-{offset + index}" for index in range(rows)],
        "plume_ids": [
            f"plume-{offset + index}" for index in range(rows)
        ],
        "event_ids": [
            f"event-{offset + index}" for index in range(rows)
        ],
        "availability_signatures": [
            "s2+l89+emit+s5p" for _ in range(rows)
        ],
        "sensor_names": ["s2", "l89", "emit", "s5p"],
        "manifest": {
            "path": f"/synthetic/{split}.csv",
            "sha256": f"synthetic-{split}",
            "rows": rows,
        },
        "encoder": {"state_sha256": "same-encoder"},
    }
    torch.save(payload, path)


class StrictRunnerTest(unittest.TestCase):
    @staticmethod
    def _base_manifest_row(
        *, sample_id: str, query360_index: int, label: int
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "id": sample_id,
            "plume_id": f"plume-{sample_id}",
            "label": label,
            "cluster_id": f"cluster-{sample_id}",
            "macro_region_id": f"region-{sample_id}",
            "query360_index": query360_index,
        }
        for column in CLASSIFICATION_PATH_COLUMNS:
            row[column] = ""
        return row

    def test_declared_s5p_roles_survive_strict_finite_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="legacy_roles_") as temporary:
            root = Path(temporary)
            row = self._base_manifest_row(
                sample_id="s5p-only", query360_index=17, label=1
            )
            ch4 = np.full((3, 4, 5), np.nan, dtype=np.float32)
            ch4[0] = 1900.0
            s5p_path = root / "s5p.npz"
            np.savez(s5p_path, ch4=ch4)
            row["s5p_0_path"] = str(s5p_path)
            row["availability_signature"] = "s5p"
            manifest = root / "derived_train.csv"
            pd.DataFrame([row]).to_csv(manifest, index=False)

            dataset = runner.LegacyAllRolesDataset(
                manifest, pad_to_multiple=None
            )
            item = dataset[0]
            self.assertEqual(item["valid_mask"][3].tolist(), [True, False, False])
            self.assertEqual(
                [entry["role"] for entry in item["observations"]["s5p"]],
                [0],
            )
            declared = item["declared_observations"]["s5p"]
            self.assertEqual(
                [entry["role"] for entry in declared], [0, 1, 2]
            )
            for role in (1, 2):
                expected = -float(dataset._s5p_mean[role]) / float(
                    dataset._s5p_std[role]
                )
                self.assertTrue(
                    torch.allclose(
                        declared[role]["image"],
                        torch.full_like(declared[role]["image"], expected),
                    )
                )

            collated = runner.query360_collate([item])
            strict_batch = collated["sensor_batches"]["s5p"]
            declared_batch = collated["declared_sensor_batches"]["s5p"]
            self.assertEqual(strict_batch["roles"].tolist(), [0])
            self.assertEqual(declared_batch["roles"].tolist(), [0, 1, 2])
            legacy = collated["legacy_concat_batch"]
            self.assertEqual(legacy["sensors"], ["s5p"])
            self.assertEqual(legacy["rows"].tolist(), [17])
            self.assertEqual(tuple(legacy["images"].shape), (1, 3, 4, 5))
            self.assertEqual(tuple(legacy["channel_ids"].shape), (1, 3))

    def test_legacy_concat_globally_pads_channels_and_space(self) -> None:
        with tempfile.TemporaryDirectory(prefix="legacy_pad_") as temporary:
            root = Path(temporary)
            s5p_row = self._base_manifest_row(
                sample_id="s5p", query360_index=1, label=0
            )
            s5p_path = root / "s5p.npz"
            np.savez(
                s5p_path,
                ch4=np.full((3, 4, 5), 1900.0, dtype=np.float32),
            )
            s5p_row["s5p_0_path"] = str(s5p_path)
            s5p_row["availability_signature"] = "s5p"

            s2_row = self._base_manifest_row(
                sample_id="s2", query360_index=2, label=1
            )
            for suffix in ("0", "90", "360"):
                path = root / f"s2_{suffix}.tif"
                tifffile.imwrite(
                    path, np.ones((12, 8, 9), dtype=np.float32)
                )
                s2_row[f"s2_{suffix}_path"] = str(path)
            s2_row["availability_signature"] = "s2"
            manifest = root / "derived_train.csv"
            pd.DataFrame([s5p_row, s2_row]).to_csv(manifest, index=False)

            dataset = runner.LegacyAllRolesDataset(
                manifest, pad_to_multiple=None
            )
            collated = runner.query360_collate([dataset[0], dataset[1]])
            legacy = collated["legacy_concat_batch"]
            self.assertEqual(legacy["sensors"], ["s5p", "s2"])
            self.assertEqual(legacy["rows"].tolist(), [1, 2])
            self.assertEqual(tuple(legacy["images"].shape), (2, 36, 8, 9))
            self.assertEqual(tuple(legacy["channel_ids"].shape), (2, 36))
            self.assertEqual(
                int(torch.count_nonzero(legacy["images"][0, 3:])), 0
            )
            self.assertEqual(
                int(torch.count_nonzero(legacy["channel_ids"][0, 3:])), 0
            )
            historical_dataset = TriSensorTemporalCsvDataset(
                csv_path=str(manifest), pad_to_multiple=None
            )
            old_x, _old_y, old_sensors, old_rows = custom_collate_fn(
                [historical_dataset[0], historical_dataset[1]]
            )
            self.assertEqual(old_sensors, legacy["sensors"])
            self.assertEqual(old_rows.tolist(), [0, 1])
            self.assertTrue(torch.equal(old_x["imgs"], legacy["images"]))
            self.assertTrue(
                torch.equal(
                    old_x["chn_ids"].reshape(2, -1).to(
                        legacy["channel_ids"].dtype
                    ),
                    legacy["channel_ids"],
                )
            )

    def test_fusion_diagnostic_distinguishes_missing_declared_row(self) -> None:
        bank = runner.SensorEncoderBank.__new__(runner.SensorEncoderBank)
        torch.nn.Module.__init__(bank)
        bank.fusion_head = runner.BinaryCLSHead(embed_dim=4)
        bank.fusion_mode = "max"
        features = torch.zeros(2, 4, 4)
        valid = torch.tensor(
            [[True, False, False, False], [False, False, False, False]]
        )
        with self.assertRaisesRegex(
            RuntimeError, "query360_index values=\\[102\\]"
        ):
            bank.fuse_universal_base(
                features,
                valid,
                row_indices=torch.tensor([101, 102]),
            )

    def test_heldout_manifest_requires_explicit_dataset_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "sealed_test_manifest.csv"
            row = {
                "id": "one",
                "plume_id": "event-A",
                "label": 1,
                "cluster_id": "cluster",
                "macro_region_id": "region",
            }
            for column in CLASSIFICATION_PATH_COLUMNS:
                row[column] = ""
            row["s2_0_path"] = str(root / "s2_0.tif")
            row["s2_90_path"] = str(root / "s2_90.tif")
            row["s2_360_path"] = str(root / "s2_360.tif")
            pd.DataFrame([row]).to_csv(manifest, index=False)
            with self.assertRaises(UnsafePathError):
                Query360Dataset(manifest)
            dataset = Query360Dataset(
                manifest, allow_heldout_manifest=True
            )
            self.assertEqual(len(dataset), 1)

    def test_test_cache_is_locked_and_evaluated_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.pt"
            dev = root / "dev.pt"
            test = root / "sealed_test.pt"
            _cache(train, split="train_core", offset=0)
            _cache(dev, split="dev", offset=100)
            _cache(test, split="test", offset=200)

            fit_output = root / "fit"
            fit = runner.build_parser().parse_args(
                [
                    "train",
                    "--train-cache",
                    str(train),
                    "--dev-cache",
                    str(dev),
                    "--output-dir",
                    str(fit_output),
                    "--epochs",
                    "0",
                    "--model-dim",
                    "8",
                    "--num-heads",
                    "2",
                    "--temporal-depth",
                    "1",
                    "--batch-size",
                    "4",
                    "--eval-batch-size",
                    "4",
                    "--device",
                    "cpu",
                ]
            )
            fit.function(fit)
            fit_status = json.loads(
                (fit_output / "run_status.json").read_text()
            )
            self.assertFalse(fit_status["sealed_test_read"])
            self.assertFalse(
                (fit_output / "sealed_test_result.json").exists()
            )

            denied_output = root / "denied"
            denied = runner.build_parser().parse_args(
                [
                    "evaluate-locked",
                    "--checkpoint",
                    str(fit_output / "checkpoint_best.pth"),
                    "--selection-lock",
                    str(fit_output / "selection_lock.json"),
                    "--test-cache",
                    str(test),
                    "--output-dir",
                    str(denied_output),
                    "--device",
                    "cpu",
                ]
            )
            with self.assertRaises(PermissionError):
                denied.function(denied)
            self.assertFalse(denied_output.exists())

            output = root / "authorized"
            allowed = runner.build_parser().parse_args(
                [
                    "evaluate-locked",
                    "--checkpoint",
                    str(fit_output / "checkpoint_best.pth"),
                    "--selection-lock",
                    str(fit_output / "selection_lock.json"),
                    "--test-cache",
                    str(test),
                    "--sealed-test",
                    "--output-dir",
                    str(output),
                    "--eval-batch-size",
                    "4",
                    "--device",
                    "cpu",
                ]
            )
            allowed.function(allowed)
            lock = json.loads(
                (fit_output / "selection_lock.json").read_text()
            )
            result = json.loads(
                (output / "sealed_test_result.json").read_text()
            )
            status = json.loads(
                (output / "locked_eval_status.json").read_text()
            )
            self.assertFalse(lock["test_cache_read_before_lock"])
            self.assertEqual(result["evaluation_count"], 1)
            self.assertFalse(result["test_threshold_search_performed"])
            self.assertEqual(
                result["metrics"]["decision_threshold"],
                lock["locked_threshold"],
            )
            self.assertTrue(status["sealed_test_read"])
            self.assertEqual(status["sealed_test_evaluations"], 1)
            repeated = runner.build_parser().parse_args(
                [
                    "evaluate-locked",
                    "--checkpoint",
                    str(fit_output / "checkpoint_best.pth"),
                    "--selection-lock",
                    str(fit_output / "selection_lock.json"),
                    "--test-cache",
                    str(test),
                    "--sealed-test",
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                ]
            )
            with self.assertRaises(FileExistsError):
                repeated.function(repeated)


if __name__ == "__main__":
    unittest.main()
