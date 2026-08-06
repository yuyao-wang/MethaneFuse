#!/usr/bin/env python3
"""CPU-only regression tests for strict Query360 data handling."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import query360_data as data


def _fake_path(sample_id: str, sensor: str, role: str) -> str:
    suffix = ".npz" if sensor == "s5p" else ".tif"
    return f"/synthetic_q360/{sample_id}/{sensor}_{role}{suffix}"


def _split_row(region: int, label: int, variant: str = "a") -> dict[str, object]:
    sample_id = f"r{region}_y{label}_{variant}"
    row: dict[str, object] = {
        "id": sample_id,
        "plume_id": f"plume_r{region}_y{label}",
        "label": label,
        "cluster_id": f"cluster_r{region}_y{label}",
        "macro_region_id": region,
    }
    for sensor in ("s2", "l89", "emit"):
        for role in ("0", "90", "360"):
            row[f"{sensor}_{role}_path"] = _fake_path(
                sample_id, sensor, role
            )
    row["s5p_0_path"] = _fake_path(sample_id, "s5p", "0")
    # Mask columns are intentionally present and must never enter cache lists.
    for sensor in ("s2", "l89", "emit", "s5p"):
        row[f"{sensor}_plume_path"] = f"/unused_masks/{sample_id}_{sensor}.tif"
    return row


class Query360DataTests(unittest.TestCase):
    def test_01_path_guard_rejects_held_out_names(self) -> None:
        for unsafe in (
            "/safe/manifest_time_test.csv",
            "/safe/SEALED-v2/file.csv",
            "/safe/my_holdout_split/file.csv",
        ):
            with self.assertRaises(data.UnsafePathError):
                data.assert_safe_path(unsafe)
        self.assertEqual(
            data.assert_safe_path("/safe/training/file.csv"),
            Path("/safe/training/file.csv"),
        )

    def test_02_deterministic_region_split_is_group_and_path_disjoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_") as temporary:
            root = Path(temporary)
            rows: list[dict[str, object]] = []
            # A deliberately large region zero must remain in train.
            for label in (0, 1):
                for copy_index in range(8):
                    row = _split_row(0, label, variant=f"u{copy_index}")
                    row["plume_id"] = f"zero_plume_{label}_{copy_index}"
                    row["cluster_id"] = f"zero_cluster_{label}_{copy_index}"
                    rows.append(row)
            for region in range(1, 20):
                for label in (0, 1):
                    rows.append(_split_row(region, label))
            # Same (plume,label,signature): exactly one hash-selected row survives.
            duplicate = _split_row(1, 0, variant="duplicate")
            duplicate["plume_id"] = "plume_r1_y0"
            duplicate["cluster_id"] = "cluster_r1_y0"
            rows.append(duplicate)
            source = root / "manifest_train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)

            artifacts = data.derive_inner_split(
                source,
                root / "derived_train.csv",
                root / "inner_val.csv",
                root / "audit.json",
                seed=77,
                target_val_fraction=0.15,
                min_val_regions=8,
                search_trials=512,
            )
            train = pd.read_csv(artifacts.train_csv, dtype=str)
            inner_val = pd.read_csv(artifacts.inner_val_csv, dtype=str)
            self.assertNotIn("0", set(inner_val["macro_region_id"]))
            self.assertIn("0", set(train["macro_region_id"]))
            self.assertGreaterEqual(inner_val["macro_region_id"].nunique(), 8)
            for column in ("plume_id", "cluster_id", "macro_region_id"):
                self.assertFalse(set(train[column]) & set(inner_val[column]))
            train_paths = set(data.collect_classification_paths(train))
            val_paths = set(data.collect_classification_paths(inner_val))
            self.assertFalse(train_paths & val_paths)
            combined = pd.concat([train, inner_val], ignore_index=True)
            self.assertFalse(
                combined.duplicated(
                    ["plume_id", "label", "availability_signature"]
                ).any()
            )
            self.assertEqual(len(combined), len(rows) - 1)
            self.assertEqual(combined["query360_index"].nunique(), len(combined))

            # A second derivation with the same seed has identical membership.
            second = data.derive_inner_split(
                source,
                root / "repeat_train.csv",
                root / "repeat_inner_val.csv",
                root / "repeat_audit.json",
                seed=77,
                target_val_fraction=0.15,
                min_val_regions=8,
                search_trials=512,
            )
            second_val = pd.read_csv(second.inner_val_csv, dtype=str)
            self.assertEqual(
                set(inner_val["query360_index"]),
                set(second_val["query360_index"]),
            )
            with open(artifacts.audit_json, "r", encoding="utf-8") as stream:
                audit = json.load(stream)
            self.assertFalse(
                audit["leakage_assertions"]["external_test_manifest_read"]
            )
            self.assertFalse(audit["plume_mask_columns_read_or_cached"])
            self.assertEqual(
                audit["leakage_assertions"]["classification_path_overlap"], 0
            )

    def test_03_split_rejects_partial_sensor_and_nontrain_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_") as temporary:
            root = Path(temporary)
            row = _split_row(0, 0)
            row["s2_90_path"] = ""
            source = root / "manifest_train.csv"
            pd.DataFrame([row]).to_csv(source, index=False)
            with self.assertRaisesRegex(data.Query360DataError, "Partial s2"):
                data.derive_inner_split(
                    source,
                    root / "out_train.csv",
                    root / "inner_val.csv",
                    root / "audit.json",
                )
            neutral = root / "manifest.csv"
            pd.DataFrame([_split_row(0, 0)]).to_csv(neutral, index=False)
            with self.assertRaises(data.UnsafePathError):
                data.derive_inner_split(
                    neutral,
                    root / "out2_train.csv",
                    root / "inner2_val.csv",
                    root / "audit2.json",
                )

    def test_04_hashed_cache_is_size_strict_and_dataset_lookup_is_local_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_") as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"0123456789")
            cache = data.StrictHashedFileCache(root / "cache")
            report = cache.warm_up([source, source], max_workers=2)
            self.assertEqual(report.requested_paths, 1)
            self.assertEqual(report.copied_paths, 1)
            destination = cache.cached_path(source)
            self.assertEqual(destination.read_bytes(), source.read_bytes())

            # require_cached computes the hash lexically and touches only dst.
            original_stat = os.stat

            def guarded_stat(path, *args, **kwargs):
                if os.path.abspath(os.fspath(path)) == os.path.abspath(source):
                    raise AssertionError("require_cached touched remote source metadata")
                return original_stat(path, *args, **kwargs)

            with mock.patch("os.stat", side_effect=guarded_stat), mock.patch.object(
                Path,
                "resolve",
                side_effect=AssertionError("Path.resolve must not be used"),
            ):
                self.assertEqual(cache.require_cached(source), str(destination))

            destination.write_bytes(b"x")
            with self.assertRaises(data.CacheIntegrityError):
                cache.ensure_local(source)

    def _write_dataset_manifest(self, root: Path) -> Path:
        s2_mean = np.asarray(data.S2_PRECOMPUTED_STATS[0], dtype=np.float32)
        s2_std = np.asarray(data.S2_PRECOMPUTED_STATS[1], dtype=np.float32)
        l89_mean = np.asarray(data.L89_PRECOMPUTED_STATS[0], dtype=np.float32)
        l89_std = np.asarray(data.L89_PRECOMPUTED_STATS[1], dtype=np.float32)
        row: dict[str, object] = {
            "id": "sample-a",
            "plume_id": "plume-a",
            "label": 1,
            "cluster_id": "cluster-a",
            "macro_region_id": 7,
            "query360_index": 42,
        }
        for role, suffix in enumerate(("0", "90", "360")):
            s2 = np.broadcast_to(
                (s2_mean + role * s2_std)[:, None, None], (12, 4, 5)
            ).copy()
            if role == 1:
                s2[0, 0, 0] = np.nan
            if role == 2:
                # Exact raw duplicate of t0 must be masked.
                s2 = np.broadcast_to(s2_mean[:, None, None], (12, 4, 5)).copy()
            s2_path = root / f"s2_{suffix}.tif"
            tifffile.imwrite(s2_path, s2)
            row[f"s2_{suffix}_path"] = str(s2_path)

            l89 = np.zeros((10, 4, 5), dtype=np.float32)
            l89[:7] = np.broadcast_to(
                (l89_mean + role * l89_std)[:, None, None], (7, 4, 5)
            )
            l89_path = root / f"l89_{suffix}.tif"
            tifffile.imwrite(l89_path, l89)
            row[f"l89_{suffix}_path"] = str(l89_path)

            emit = np.full((16, 4, 5), (role + 1) * 6553.5, dtype=np.float32)
            emit_path = root / f"emit_{suffix}.tif"
            tifffile.imwrite(emit_path, emit)
            row[f"emit_{suffix}_path"] = str(emit_path)

        s5p = np.empty((3, 4, 5), dtype=np.float32)
        for role, mean in enumerate(data.S5P_PRECOMPUTED_STATS[0]):
            s5p[role] = mean + role + np.arange(20, dtype=np.float32).reshape(4, 5)
        s5p_path = root / "s5p_0.npz"
        np.savez(s5p_path, ch4=s5p)
        row["s5p_0_path"] = str(s5p_path)
        row["availability_signature"] = "s2+l89+emit+s5p"
        manifest = root / "derived_train.csv"
        pd.DataFrame([row]).to_csv(manifest, index=False)
        return manifest

    def test_05_dataset_normalizes_masks_duplicates_and_collates_axes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_") as temporary:
            root = Path(temporary)
            manifest = self._write_dataset_manifest(root)
            dataset = data.Query360Dataset(manifest, pad_to_multiple=None)
            item = dataset[0]
            self.assertEqual(item["index"], 42)
            self.assertEqual(tuple(item["valid_mask"].shape), (4, 3))
            self.assertEqual(
                item["valid_mask"][0].tolist(), [True, True, False]
            )
            expected_fraction = (12 * 4 * 5 - 1) / (12 * 4 * 5)
            self.assertAlmostEqual(
                float(item["finite_fraction"][0, 1]), expected_fraction, places=6
            )
            s2_role1 = item["observations"]["s2"][1]["image"]
            # Invalid source pixels are neutral zero *after* normalization.
            self.assertEqual(float(s2_role1[0, 0, 0]), 0.0)
            self.assertTrue(torch.allclose(s2_role1[1:, :, :], torch.ones_like(s2_role1[1:, :, :])))
            self.assertEqual(
                [observation["role"] for observation in item["observations"]["l89"]],
                [0, 1, 2],
            )
            self.assertAlmostEqual(
                float(item["observations"]["emit"][0]["image"].mean()),
                0.1,
                places=5,
            )

            collated = data.query360_collate([item])
            self.assertEqual(collated["index"].tolist(), [42])
            self.assertEqual(collated["labels"].tolist(), [1])
            self.assertEqual(list(collated["sensor_batches"]), list(data.SENSOR_ORDER))
            self.assertEqual(
                collated["sensor_batches"]["s2"]["roles"].tolist(), [0, 1]
            )
            self.assertEqual(
                collated["sensor_batches"]["s2"]["rows"].tolist(), [42, 42]
            )
            expected_channels = {"s2": 12, "l89": 7, "emit": 16, "s5p": 1}
            for sensor, channels in expected_channels.items():
                sensor_batch = collated["sensor_batches"][sensor]
                self.assertEqual(sensor_batch["images"].shape[1], channels)
                self.assertEqual(sensor_batch["channel_ids"].shape, (channels,))
            self.assertEqual(tuple(collated["valid_mask"].shape), (1, 4, 3))
            self.assertEqual(
                tuple(collated["finite_fraction"].shape), (1, 4, 3)
            )

    def test_06_t0_below_finite_threshold_masks_entire_sensor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_") as temporary:
            root = Path(temporary)
            manifest = self._write_dataset_manifest(root)
            frame = pd.read_csv(manifest)
            t0_path = Path(frame.loc[0, "s2_0_path"])
            bad = np.full((12, 4, 5), np.nan, dtype=np.float32)
            bad[0, 0, 0] = 1.0  # 1/240 < default 5% threshold
            tifffile.imwrite(t0_path, bad)
            dataset = data.Query360Dataset(manifest, pad_to_multiple=None)
            item = dataset[0]
            self.assertEqual(item["valid_mask"][0].tolist(), [False, False, False])
            self.assertEqual(item["observations"]["s2"], [])

    def test_07_s5p_duplicate_detection_uses_raw_roles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q360_") as temporary:
            root = Path(temporary)
            manifest = self._write_dataset_manifest(root)
            frame = pd.read_csv(manifest)
            s5p_path = Path(frame.loc[0, "s5p_0_path"])
            repeated = np.full((3, 4, 5), 1900.0, dtype=np.float32)
            np.savez(s5p_path, ch4=repeated)
            dataset = data.Query360Dataset(manifest, pad_to_multiple=None)
            item = dataset[0]
            self.assertEqual(item["valid_mask"][3].tolist(), [True, False, False])
            self.assertEqual(
                [entry["role"] for entry in item["observations"]["s5p"]], [0]
            )


if __name__ == "__main__":
    unittest.main()
