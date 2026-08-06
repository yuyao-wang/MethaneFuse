#!/usr/bin/env python3
"""CPU-only tests for the strict EMIT32 frozen-CLS wrapper."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    emit_ragged_cls_experiment as runner,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as common,
)


class FakeBackbone(nn.Module):
    embed_dim = 16

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.batch_sizes: list[int] = []
        self.channel_ids_seen: list[torch.Tensor] = []

    def forward_features(self, x_dict):
        images = x_dict["imgs"]
        channel_ids = x_dict["chn_ids"]
        if images.ndim != 4 or images.shape[1] != 32:
            raise AssertionError(f"Expected flattened BCHW32, got {images.shape}.")
        expected = torch.tensor(
            runner.emit_module.EMIT32_WAVELENGTHS_NM,
            dtype=channel_ids.dtype,
            device=channel_ids.device,
        ).view(1, -1)
        if not torch.equal(channel_ids, expected.expand_as(channel_ids)):
            raise AssertionError("Physical EMIT wavelength IDs were not forwarded.")
        self.batch_sizes.append(int(images.shape[0]))
        self.channel_ids_seen.append(channel_ids.detach().cpu())
        pooled = images.mean(dim=(-2, -1))
        return {"x_norm_clstoken": pooled[:, : self.embed_dim]}


class EmitRaggedClsTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(
        root: Path,
        *,
        split: str,
        event_prefix: str,
        rows: int,
        duplicate_first_seasonal: bool = False,
    ) -> Path:
        records = []
        role_offsets = {
            "t0": 0.0,
            "prev1": 100.0,
            "seasonal": 200.0,
        }
        for row in range(rows):
            paths: dict[str, str] = {}
            for role, offset in role_offsets.items():
                array = np.empty((4, 4, 32), dtype=np.float32)
                for band in range(32):
                    array[..., band] = 1.0 + row * 10.0 + offset + band
                path = root / f"{split}-{row}-{role}.tif"
                tifffile.imwrite(path, array)
                paths[role] = str(path)
            t0_day = 10 + row
            seasonal_time = (
                f"2026-01-{t0_day:02d}T00:00:00Z"
                if duplicate_first_seasonal and row == 0
                else f"2025-10-{10 + row:02d}T00:00:00Z"
            )
            records.append(
                {
                    "sample_id": f"{split}-sample-{row}",
                    "label": row % 2,
                    "plume_id": f"{event_prefix}{row}-A",
                    "event_time": f"2026-01-{t0_day:02d}T00:00:00Z",
                    "t0_image_time": f"2026-01-{t0_day:02d}T00:00:00Z",
                    "prev1_image_time": f"2026-01-{t0_day - 1:02d}T00:00:00Z",
                    "seasonal_image_time": seasonal_time,
                    "path_t0": paths["t0"],
                    "path_prev1": paths["prev1"],
                    "path_seasonal": paths["seasonal"],
                }
            )
        path = root / f"{split}.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        return path

    @staticmethod
    def _write_bound_stats(root: Path, train_csv: Path) -> Path:
        block = {
            "frames_per_row": 3,
            "mean": [0.0] * 32,
            "std": [1.0] * 32,
            "requested_sample_rows": 8,
            "sampled_rows": 8,
            "sample_seed": 17,
            "train_csv": str(train_csv.resolve()),
            "train_csv_sha256": common.sha256_file(train_csv),
            "zero_is_nodata": True,
        }
        path = root / "normalization.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "representation": "normalized raw frames",
                    "sensors": {"emit": block},
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _cache_cli(
        *,
        csv_path: Path,
        split: str,
        output_path: Path,
        weights_path: Path,
        stats_path: Path,
        train_csv: Path,
    ) -> list[str]:
        return [
            "cache",
            "--csv",
            str(csv_path),
            "--split",
            split,
            "--output-cache",
            str(output_path),
            "--weights",
            str(weights_path),
            "--stats-json",
            str(stats_path),
            "--normalization-train-csv",
            str(train_csv),
            "--image-size",
            "4",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--amp-dtype",
            "float32",
            "--storage-dtype",
            "float32",
            "--local-cache-mode",
            "off",
            "--log-interval",
            "99",
        ]

    def test_01_stats_are_nested_and_bound_to_exact_train_sha(self) -> None:
        with tempfile.TemporaryDirectory(prefix="emit-ragged.") as directory:
            root = Path(directory)
            train_csv = self._write_manifest(
                root, split="train", event_prefix="train-event-", rows=2
            )
            stats_path = self._write_bound_stats(root, train_csv)
            mean, std, provenance = runner.load_sha_bound_emit_stats(
                stats_path, train_csv
            )
            self.assertEqual(mean, [0.0] * 32)
            self.assertEqual(std, [1.0] * 32)
            self.assertEqual(
                provenance["normalization_train_csv_sha256"],
                common.sha256_file(train_csv),
            )

            frame = pd.read_csv(train_csv)
            frame.loc[0, "label"] = 1
            frame.to_csv(train_csv, index=False)
            with self.assertRaisesRegex(ValueError, "not bound"):
                runner.load_sha_bound_emit_stats(stats_path, train_csv)

            flat_path = root / "flat.json"
            flat_path.write_text(
                json.dumps({"mean": [0.0] * 32, "std": [1.0] * 32}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sensors.emit"):
                runner.load_sha_bound_emit_stats(flat_path, train_csv)

    def test_02_hwc32_cache_preserves_wavelengths_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="emit-ragged.") as directory:
            root = Path(directory)
            train_csv = self._write_manifest(
                root,
                split="train",
                event_prefix="train-event-",
                rows=4,
                duplicate_first_seasonal=True,
            )
            stats_path = self._write_bound_stats(root, train_csv)
            weights_path = root / "weights.pth"
            weights_path.write_bytes(b"synthetic-panopticon")
            cache_path = root / "train-cache.pt"
            fake = FakeBackbone()
            with mock.patch.object(
                runner.emit_module, "load_backbone", return_value=fake
            ):
                runner.main(
                    self._cache_cli(
                        csv_path=train_csv,
                        split="train",
                        output_path=cache_path,
                        weights_path=weights_path,
                        stats_path=stats_path,
                        train_csv=train_csv,
                    )
                )
            self.assertEqual(fake.batch_sizes, [12])
            payload = common.torch_load_trusted(cache_path)
            self.assertEqual(payload["sensor"], runner.SENSOR_NAME)
            self.assertEqual(payload["script_version"], runner.SCRIPT_VERSION)
            self.assertEqual(tuple(payload["features"].shape), (4, 3, 16))
            self.assertEqual(payload["ids"][0], "train-sample-0")
            self.assertEqual(payload["event_ids"][0], "train-event-0")
            self.assertEqual(
                payload["event_id_rule"], "plume_id:strip-final-hyphen-suffix"
            )
            expected_ids = torch.tensor(
                runner.emit_module.EMIT32_WAVELENGTHS_NM,
                dtype=torch.float32,
            ).tolist()
            self.assertEqual(
                payload["input_contract"]["channel_ids"], expected_ids
            )
            self.assertEqual(
                payload["input_contract"]["channel_id_semantics"],
                "physical-wavelength-nanometers",
            )
            self.assertEqual(
                payload["input_contract"]["reader"],
                "Emit32CsvDataset:HWC-or-CHW-to-CHW-v1",
            )
            # HWC band 0 was read as CHW channel 0, with no normalization shift.
            self.assertAlmostEqual(float(payload["features"][1, 0, 0]), 11.0)
            self.assertTrue(bool(payload["duplicate_mask"][0, 2]))
            self.assertFalse(bool(payload["unique_mask"][0, 2]))
            self.assertEqual(
                int(torch.count_nonzero(payload["features"][0, 2])), 0
            )
            self.assertEqual(
                payload["input_contract"]["normalization_source"][
                    "normalization_train_csv_sha256"
                ],
                common.sha256_file(train_csv),
            )
            self.assertTrue(
                cache_path.with_suffix(cache_path.suffix + ".json").is_file()
            )

    def test_03_cli_cache_to_four_heads_and_overlap_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="emit-ragged.") as directory:
            root = Path(directory)
            train_csv = self._write_manifest(
                root, split="train", event_prefix="train-event-", rows=8
            )
            val_csv = self._write_manifest(
                root, split="val", event_prefix="val-event-", rows=8
            )
            stats_path = self._write_bound_stats(root, train_csv)
            weights_path = root / "weights.pth"
            weights_path.write_bytes(b"synthetic-panopticon")
            train_cache = root / "train-cache.pt"
            val_cache = root / "val-cache.pt"
            fake = FakeBackbone()
            with mock.patch.object(
                runner.emit_module, "load_backbone", return_value=fake
            ):
                runner.main(
                    self._cache_cli(
                        csv_path=train_csv,
                        split="train",
                        output_path=train_cache,
                        weights_path=weights_path,
                        stats_path=stats_path,
                        train_csv=train_csv,
                    )
                )
                runner.main(
                    self._cache_cli(
                        csv_path=val_csv,
                        split="val",
                        output_path=val_cache,
                        weights_path=weights_path,
                        stats_path=stats_path,
                        train_csv=train_csv,
                    )
                )
            output_dir = root / "heads"
            runner.main(
                [
                    "train-heads",
                    "--train-cache",
                    str(train_cache),
                    "--val-cache",
                    str(val_cache),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--eval-batch-size",
                    "4",
                    "--model-dim",
                    "16",
                    "--num-heads",
                    "4",
                    "--dropout",
                    "0",
                    "--max-train-steps",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "23",
                ]
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["script_version"], runner.SCRIPT_VERSION)
            self.assertEqual(set(summary["best_by_arm"]), set(runner.ARM_NAMES))
            self.assertEqual(
                summary["cache_audit"]["sensor"], runner.SENSOR_NAME
            )
            self.assertEqual(summary["cache_audit"]["event_overlap"], 0)
            self.assertFalse(summary["sealed_test_read"])
            for arm in runner.ARM_NAMES:
                metrics = summary["best_by_arm"][arm]["validation"]
                for key in (
                    "ap",
                    "auc",
                    "macro_f1_at_0_5",
                    "balanced_accuracy_at_0_5",
                    "pred_positive_rate_at_0_5",
                ):
                    self.assertIn(key, metrics)
                self.assertTrue(
                    (output_dir / arm / "checkpoint_best_ap.pt").is_file()
                )

            overlap_cache = root / "val-overlap.pt"
            overlapping = common.torch_load_trusted(val_cache)
            train_payload = common.torch_load_trusted(train_cache)
            overlapping["event_ids"][0] = train_payload["event_ids"][0]
            common.atomic_torch_save(overlap_cache, overlapping)
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.main(
                    [
                        "train-heads",
                        "--train-cache",
                        str(train_cache),
                        "--val-cache",
                        str(overlap_cache),
                        "--output-dir",
                        str(root / "overlap-heads"),
                        "--epochs",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )

    def test_04_sealed_or_test_like_paths_are_refused_before_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="emit-ragged.") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "sealed/test"):
                runner.main(
                    [
                        "cache",
                        "--csv",
                        str(root / "sealed" / "train.csv"),
                        "--split",
                        "train",
                        "--output-cache",
                        str(root / "cache.pt"),
                    ]
                )
            with self.assertRaisesRegex(ValueError, "sealed/test"):
                runner.main(
                    [
                        "cache",
                        "--csv",
                        str(root / "train.csv"),
                        "--split",
                        "train",
                        "--output-cache",
                        str(root / "test-output" / "cache.pt"),
                    ]
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
