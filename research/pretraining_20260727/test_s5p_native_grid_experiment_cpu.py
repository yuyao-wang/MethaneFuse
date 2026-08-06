#!/usr/bin/env python3
"""Synthetic CPU tests for the S5P approximate native-grid experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (
    s5p_native_grid_experiment as experiment,
)


class S5PNativeGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_01_finite_weighted_pooling_and_valid_mask(self) -> None:
        field = torch.full((1, 6, 6), 4.0)
        field[:, 0:2, 0:2] = torch.tensor(
            [[[1.0, float("nan")], [3.0, float("nan")]]]
        )
        field[:, 0:2, 2:4] = float("nan")
        field[:, 0:2, 4:6] = 0.0

        pooled, valid = experiment.finite_mask_weighted_adaptive_pool(
            field, (3, 3)
        )

        self.assertEqual(tuple(pooled.shape), (1, 3, 3))
        self.assertEqual(tuple(valid.shape), (1, 3, 3))
        self.assertEqual(float(pooled[0, 0, 0]), 2.0)
        self.assertTrue(bool(valid[0, 0, 0]))
        self.assertEqual(float(pooled[0, 0, 1]), 0.0)
        self.assertFalse(bool(valid[0, 0, 1]))
        self.assertEqual(float(pooled[0, 0, 2]), 0.0)
        self.assertTrue(
            bool(valid[0, 0, 2]),
            "Finite S5P zero must remain valid",
        )
        self.assertTrue(torch.isfinite(pooled).all())

    def test_02_utc_delta_days_and_fallback(self) -> None:
        record = {
            "_row_id": 7,
            "plume_time": "2025-01-02T12:00:00+00:00",
            "t0_path": "S5P_OFFL_20250102T120000_x.nc",
            "prev1_path": "S5P_OFFL_20250101T000000_x.nc",
            "prev2_path": "S5P_OFFL_20241231T120000_x.nc",
            "prev3_path": "unparseable.nc",
            "seasonal_path": "S5P_OFFL_20241004T120000_x.nc",
            "year_path": "S5P_OFFL_20240103T120000_x.nc",
        }
        deltas, fallback = experiment.utc_delta_days_from_record(record)

        expected = torch.tensor([0.0, -1.5, -2.0, -3.0, -90.0, -365.0])
        self.assertTrue(torch.equal(deltas, expected))
        self.assertTrue(
            torch.equal(
                fallback,
                torch.tensor([False, False, False, True, False, False]),
            )
        )

    def test_03_history_shuffle_preserves_t0_and_uses_donors(self) -> None:
        features = torch.arange(
            3 * 6 * 3 * 3, dtype=torch.float32
        ).reshape(3, 6, 3, 3)
        valid = (features.remainder(4) != 0)
        permutation = torch.tensor([1, 2, 0])

        shuffled, shuffled_valid = experiment.shuffle_history_only(
            features, valid, permutation
        )

        self.assertTrue(torch.equal(shuffled[:, 0], features[:, 0]))
        self.assertTrue(torch.equal(shuffled_valid[:, 0], valid[:, 0]))
        self.assertTrue(
            torch.equal(shuffled[:, 1:], features[permutation, 1:])
        )
        self.assertTrue(
            torch.equal(
                shuffled_valid[:, 1:], valid[permutation, 1:]
            )
        )

        event_ids = ["a", "a", "b", "b", "c", "c", "d"]
        generated = experiment.build_cross_event_donor_indices(
            event_ids, seed=11
        )
        self.assertTrue(
            all(
                event_ids[index] != event_ids[donor]
                for index, donor in enumerate(generated.tolist())
            )
        )

    def test_04_arm_mask_and_metadata_semantics(self) -> None:
        features = torch.randn(4, 6, 3, 3)
        valid = torch.ones_like(features, dtype=torch.bool)
        delta = torch.tensor(
            [[0.0, -1.0, -2.0, -3.0, -90.0, -365.0]] * 4
        )
        roles = torch.arange(6).repeat(4, 1)

        t0 = experiment.apply_arm_inputs(
            "t0_masked",
            features,
            valid,
            delta,
            roles,
            training=True,
        )
        self.assertTrue(torch.equal(t0.features[:, 0], features[:, 0]))
        self.assertTrue(torch.equal(t0.valid_mask[:, 0], valid[:, 0]))
        self.assertEqual(int(t0.valid_mask[:, 1:].sum()), 0)
        self.assertEqual(int(torch.count_nonzero(t0.features[:, 1:])), 0)
        self.assertFalse(t0.use_delta)

        role_only = experiment.apply_arm_inputs(
            "role_only",
            features,
            valid,
            delta,
            roles,
            training=False,
        )
        self.assertIs(role_only.features, features)
        self.assertFalse(role_only.use_delta)
        delta_time = experiment.apply_arm_inputs(
            "delta_time",
            features,
            valid,
            delta,
            roles,
            training=False,
        )
        self.assertTrue(delta_time.use_delta)
        shuffled_eval = experiment.apply_arm_inputs(
            "history_shuffle_train",
            features,
            valid,
            delta,
            roles,
            training=False,
        )
        self.assertIs(shuffled_eval.features, features)
        self.assertIs(shuffled_eval.delta_days, delta)

    def test_05_t0_masked_equals_legacy_t0_only_forward(self) -> None:
        torch.manual_seed(17)
        model = experiment.NativeGridTemporalClassifier(
            train_mean=0.0,
            train_std=1.0,
            hidden_dim=16,
            num_heads=4,
            dropout=0.0,
        )
        model.eval()
        features = torch.randn(3, 6, 3, 3)
        valid = torch.ones_like(features, dtype=torch.bool)
        delta = torch.tensor(
            [[0.0, -1.0, -2.0, -3.0, -90.0, -365.0]] * 3
        )
        roles = torch.arange(6).repeat(3, 1)
        masked = experiment.apply_arm_inputs(
            "t0_masked",
            features,
            valid,
            delta,
            roles,
            training=False,
        )

        with torch.inference_mode():
            masked_logits = model(
                masked.features,
                masked.valid_mask,
                masked.delta_days,
                masked.roles,
                use_delta=False,
            )
            legacy_logits = model(
                features[:, :1],
                valid[:, :1],
                delta[:, :1],
                roles[:, :1],
                use_delta=False,
            )
        self.assertTrue(
            torch.allclose(
                masked_logits, legacy_logits, atol=1e-6, rtol=0
            ),
            f"masked={masked_logits}, legacy={legacy_logits}",
        )

    def test_06_atomic_cache_contains_input_sha_and_row_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s5p_native_cache.") as temp:
            root = Path(temp)
            npz_root = root / "npz"
            npz_root.mkdir()
            array = np.arange(
                6 * 224 * 224, dtype=np.float32
            ).reshape(6, 224, 224)
            array[0, :20, :20] = np.nan
            npz_path = npz_root / "sample.npz"
            np.savez_compressed(npz_path, ch4=array)
            record = {
                "_row_id": 42,
                "image_path": str(npz_path),
                "label": 1,
                "plume_id": "plume-42",
                "plume_time": "2025-01-02T12:00:00+00:00",
            }
            for role, days in experiment.FALLBACK_OFFSETS_DAYS.items():
                timestamp = pd.Timestamp(record["plume_time"]) + pd.Timedelta(
                    days=days
                )
                record[f"{role}_path"] = (
                    f"S5P_{timestamp.strftime('%Y%m%dT%H%M%S')}_x.nc"
                )
            frame = pd.DataFrame([record])

            payload, cache_path = experiment.build_or_load_feature_cache(
                "synthetic",
                frame,
                csv_sha256="a" * 64,
                selected_sha256="b" * 64,
                local_npz_root=npz_root,
                cache_dir=root / "cache",
                data_key="ch4",
                workers=1,
                progress_every=0,
                rebuild=False,
            )

            self.assertTrue(cache_path.is_file())
            self.assertTrue(
                cache_path.with_suffix(".meta.json").is_file()
            )
            self.assertEqual(payload["row_ids"].tolist(), [42])
            self.assertEqual(len(payload["meta"]["input_sha256"]), 64)
            self.assertIn(
                "not exact original/native",
                payload["meta"]["native_grid_disclaimer"],
            )
            self.assertFalse(
                any(cache_path.parent.glob("*.part.*"))
            )

    @staticmethod
    def _synthetic_cache(seed: int, rows: int) -> dict:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(rows).remainder(2).long()
        features = torch.randn(
            rows, 6, 3, 3, generator=generator
        )
        features[:, 0] += labels[:, None, None].float() * 0.5
        valid = torch.ones_like(features, dtype=torch.bool)
        valid[::4, 4, 0, 0] = False
        delta = torch.tensor(
            [[0.0, -1.0, -2.0, -3.0, -90.0, -365.0]] * rows
        )
        return {
            "features": features,
            "valid_mask": valid,
            "delta_days": delta,
            "roles": torch.arange(6).repeat(rows, 1),
            "labels": labels,
            "row_ids": torch.arange(rows),
            "plume_ids": [f"p-{index}" for index in range(rows)],
            "event_ids": [f"event-{index // 2}" for index in range(rows)],
            "timestamp_fallback_mask": torch.zeros(
                rows, 6, dtype=torch.bool
            ),
            "meta": {"input_sha256": f"{seed:064x}"[-64:]},
        }

    def test_07_tiny_one_epoch_all_matched_arms(self) -> None:
        train = self._synthetic_cache(31, 12)
        val = self._synthetic_cache(37, 8)
        normalization = experiment.training_normalization(train)
        initial_hashes = set()
        with tempfile.TemporaryDirectory(prefix="s5p_native_train.") as temp:
            for arm in experiment.ARM_NAMES:
                result = experiment.train_one_arm(
                    arm,
                    train,
                    val,
                    normalization=normalization,
                    epochs=1,
                    batch_size=4,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    hidden_dim=16,
                    num_heads=4,
                    dropout=0.0,
                    seed=101,
                    device=torch.device("cpu"),
                    checkpoint_path=Path(temp) / f"{arm}.pt",
                )
                initial_hashes.add(result["initial_state_sha256"])
                metrics = result["best"]["val"]
                for key in (
                    "ap",
                    "auc",
                    "macro_f1",
                    "balanced_accuracy",
                    "pred_positive_rate",
                    "best_threshold",
                ):
                    self.assertIn(key, metrics)
                self.assertTrue(Path(result["checkpoint"]).is_file())
        self.assertEqual(len(initial_hashes), 1)


if __name__ == "__main__":
    unittest.main()
