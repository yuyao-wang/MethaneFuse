#!/usr/bin/env python3
"""CPU-only synthetic regression tests for the L89 Ragged6-Delta runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import json
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import l89_ragged_cls_experiment as runner


class L89RaggedClsTests(unittest.TestCase):
    def test_01_utc_delta_days_uses_real_t0(self) -> None:
        frame = pd.DataFrame(
            {
                "t0_image_time": [
                    "2026-01-10T12:00:00-05:00",
                    "2026-03-01T00:00:00Z",
                ],
                "prev1_image_time": [
                    "2026-01-08T17:00:00Z",
                    "2026-02-28T12:00:00Z",
                ],
                "seasonal_image_time": [
                    "2025-10-12T17:00:00Z",
                    None,
                ],
            }
        )
        metadata = runner.build_temporal_metadata(
            frame,
            ("path_t0", "path_prev1", "path_seasonal"),
            ("t0_image_time", "prev1_image_time", "seasonal_image_time"),
        )
        expected = torch.tensor([[0.0, -2.0, -90.0], [0.0, -0.5, float("nan")]])
        self.assertTrue(
            torch.allclose(
                torch.nan_to_num(metadata.delta_days),
                torch.nan_to_num(expected),
                atol=0,
                rtol=0,
            )
        )
        self.assertTrue(torch.isnan(metadata.delta_days[1, 2]))
        self.assertEqual(metadata.timestamps_utc_iso[0][0], "2026-01-10T17:00:00Z")
        self.assertEqual(metadata.t0_index, 0)

    def test_02_duplicate_mask_retains_t0_then_stable_role(self) -> None:
        timestamps = torch.tensor(
            [
                [100, 100, 90, 90, 80, runner.NAT_INT64],
                [100, 100, 90, 80, 70, 60],
            ],
            dtype=torch.int64,
        )
        valid = torch.tensor(
            [
                [True, True, True, True, True, False],
                [False, True, True, True, True, True],
            ]
        )
        duplicate, unique, group = runner.compute_duplicate_unique_masks(
            timestamps, valid, t0_index=0
        )
        self.assertEqual(
            unique[0].tolist(), [True, False, True, False, True, False]
        )
        self.assertEqual(
            duplicate[0].tolist(), [False, True, False, True, False, False]
        )
        self.assertEqual(
            group[0].tolist(), [True, True, True, True, False, False]
        )
        # Invalid t0 cannot suppress a valid duplicate observation.
        self.assertFalse(bool(unique[1, 0]))
        self.assertTrue(bool(unique[1, 1]))

    def test_03_all_arms_have_identical_parameter_shapes(self) -> None:
        signatures = []
        state_shapes = []
        for _arm in runner.ARM_NAMES:
            model = runner.RaggedCurrentQueryHead(
                feature_dim=16,
                num_roles=6,
                model_dim=32,
                num_heads=4,
                depth=2,
                dropout=0.0,
                t0_index=0,
            )
            signatures.append(runner.model_parameter_signature(model))
            state_shapes.append(
                {name: tuple(value.shape) for name, value in model.state_dict().items()}
            )
        for signature in signatures[1:]:
            self.assertEqual(signature, signatures[0])
        for shapes in state_shapes[1:]:
            self.assertEqual(shapes, state_shapes[0])
        self.assertEqual(len(signatures[0]["parameter_shapes"]), 41)

    def test_04_history_shuffle_is_cross_event_and_preserves_t0_labels(self) -> None:
        events = ["A", "A", "B", "C", "C"]
        donors = runner.build_cross_event_donor_indices(events, seed=19)
        for row_index, donor_index in enumerate(donors.tolist()):
            self.assertNotEqual(events[row_index], events[donor_index])

        features = torch.arange(5 * 3 * 2, dtype=torch.float32).reshape(5, 3, 2)
        valid = torch.ones(5, 3, dtype=torch.bool)
        unique = torch.ones(5, 3, dtype=torch.bool)
        delta = torch.tensor([[0.0, -1.0, -90.0]]).repeat(5, 1)
        labels = torch.tensor([0, 1, 0, 1, 1])
        original_labels = labels.clone()
        shuffled = runner.apply_history_donors(
            features,
            valid,
            unique,
            delta,
            donors,
            t0_index=0,
        )
        shuffled_features, shuffled_valid, shuffled_unique, shuffled_delta = shuffled
        self.assertTrue(torch.equal(shuffled_features[:, 0], features[:, 0]))
        self.assertTrue(torch.equal(shuffled_valid[:, 0], valid[:, 0]))
        self.assertTrue(torch.equal(shuffled_unique[:, 0], unique[:, 0]))
        self.assertTrue(torch.equal(shuffled_delta[:, 0], delta[:, 0]))
        self.assertTrue(
            torch.equal(shuffled_features[:, 1:], features[donors][:, 1:])
        )
        self.assertTrue(torch.equal(labels, original_labels))

    def test_05_t0_arm_keeps_static_tokens_but_zeros_history_evidence(self) -> None:
        features = torch.randn(4, 6, 8)
        original = features.clone()
        valid = torch.ones(4, 6, dtype=torch.bool)
        unique = torch.ones(4, 6, dtype=torch.bool)
        delta = torch.tensor([[0.0, -5.0, -10.0, -15.0, -90.0, -365.0]]).repeat(
            4, 1
        )
        prepared_features, prepared_valid, prepared_delta, enable_delta = (
            runner.prepare_arm_inputs(
                features,
                valid,
                unique,
                delta,
                arm="t0_masked",
                t0_index=0,
            )
        )
        self.assertEqual(prepared_features.shape, original.shape)
        self.assertTrue(torch.equal(prepared_features[:, 0], original[:, 0]))
        self.assertEqual(int(torch.count_nonzero(prepared_features[:, 1:])), 0)
        self.assertTrue(prepared_valid[:, 0].all())
        self.assertFalse(prepared_valid[:, 1:].any())
        self.assertEqual(int(torch.count_nonzero(prepared_delta[:, 1:])), 0)
        self.assertFalse(enable_delta)

    def test_06_every_arm_forward_backward_is_finite(self) -> None:
        torch.manual_seed(7)
        rows, timepoints, feature_dim = 6, 3, 16
        features = torch.randn(rows, timepoints, feature_dim)
        valid = torch.ones(rows, timepoints, dtype=torch.bool)
        unique = torch.ones(rows, timepoints, dtype=torch.bool)
        delta = torch.tensor([[0.0, -3.0, -90.0]]).repeat(rows, 1)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        events = [f"event-{index}" for index in range(rows)]
        donors = runner.build_cross_event_donor_indices(events, seed=3)
        role_index = torch.arange(timepoints)

        for arm in runner.ARM_NAMES:
            model = runner.RaggedCurrentQueryHead(
                feature_dim=feature_dim,
                num_roles=timepoints,
                model_dim=32,
                num_heads=4,
                depth=2,
                dropout=0.0,
                t0_index=0,
            )
            prepared = runner.prepare_arm_inputs(
                features,
                valid,
                unique,
                delta,
                arm=arm,
                t0_index=0,
                donor_indices=donors if arm == "history_shuffle_train" else None,
            )
            logits = model(
                prepared[0],
                prepared[1],
                role_index,
                prepared[2],
                enable_delta=prepared[3],
            )
            self.assertEqual(tuple(logits.shape), (rows,))
            self.assertTrue(torch.isfinite(logits).all())
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.classifier.weight.grad)
            self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
            self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0.0)

    @staticmethod
    def _synthetic_cache(split: str, *, event_prefix: str) -> dict:
        rows, timepoints, feature_dim = 8, 3, 16
        generator = torch.Generator().manual_seed(11 if split == "train" else 13)
        features = torch.randn(rows, timepoints, feature_dim, generator=generator)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        timestamps = torch.tensor(
            [[1000 + row, 999 + row, 910 + row] for row in range(rows)],
            dtype=torch.int64,
        )
        delta = torch.tensor([[0.0, -1.0, -90.0]]).repeat(rows, 1)
        valid = torch.ones(rows, timepoints, dtype=torch.bool)
        unique = torch.ones_like(valid)
        duplicate = torch.zeros_like(valid)
        contract = {
            "script_version": runner.SCRIPT_VERSION,
            "csv_sha256": f"{split}-csv",
            "weights_sha256": "a" * 64,
            "input_table_sha256": f"{split}-table",
            "path_columns": ["path_t0", "path_prev1", "path_seasonal"],
            "time_columns": [
                "t0_image_time",
                "prev1_image_time",
                "seasonal_image_time",
            ],
            "role_names": ["t0", "prev1", "seasonal"],
            "band_indices": list(range(7)),
            "channel_ids": [1.0] * 7,
            "normalization_mean": [0.0] * 7,
            "normalization_std": [1.0] * 7,
            "normalization_source": "synthetic",
            "image_size": 4,
            "min_valid_fraction": 0.75,
            "validity_band_index": 0,
            "zero_invalid_pixels": True,
            "duplicate_rule": "synthetic",
        }
        return {
            "format_version": runner.CACHE_FORMAT_VERSION,
            "script_version": runner.SCRIPT_VERSION,
            "split": split,
            "features": features,
            "labels": labels,
            "ids": [f"{split}-id-{row}" for row in range(rows)],
            "plume_ids": [f"{event_prefix}{row}-A" for row in range(rows)],
            "event_ids": [f"{event_prefix}{row}" for row in range(rows)],
            "event_id_rule": "synthetic",
            "timestamps_utc_ns": timestamps,
            "timestamps_utc_iso": [["", "", ""] for _ in range(rows)],
            "timestamp_valid_mask": valid,
            "delta_days": delta,
            "role_names": ["t0", "prev1", "seasonal"],
            "role_index": torch.arange(timepoints),
            "t0_index": 0,
            "image_valid_mask": valid,
            "valid_mask": valid,
            "duplicate_mask": duplicate,
            "duplicate_group_mask": duplicate,
            "unique_mask": unique,
            "valid_fraction": torch.ones(rows, timepoints),
            "load_status": torch.ones(rows, timepoints, dtype=torch.int8),
            "path_columns": ["path_t0", "path_prev1", "path_seasonal"],
            "time_columns": [
                "t0_image_time",
                "prev1_image_time",
                "seasonal_image_time",
            ],
            "input_contract": contract,
            "input_contract_sha256": runner.sha256_bytes(
                runner.canonical_json_bytes(contract)
            ),
            "csv_path": f"/synthetic/{split}.csv",
            "csv_sha256": f"{split}-csv",
            "weights_path": "/synthetic/weights.pt",
            "weights_sha256": "a" * 64,
            "input_table_sha256": f"{split}-table",
            "feature_sha256": runner.tensor_sha256(features),
            "label_sha256": runner.tensor_sha256(labels),
            "timestamp_sha256": runner.tensor_sha256(timestamps),
        }

    def test_07_synthetic_cache_to_four_head_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-heads.") as directory:
            root = Path(directory)
            train_path = root / "train.pt"
            val_path = root / "val.pt"
            output_dir = root / "heads"
            runner.atomic_torch_save(
                train_path, self._synthetic_cache("train", event_prefix="train-event-")
            )
            runner.atomic_torch_save(
                val_path, self._synthetic_cache("val", event_prefix="val-event-")
            )
            args = Namespace(
                train_cache=str(train_path),
                val_cache=str(val_path),
                output_dir=str(output_dir),
                arms=",".join(runner.ARM_NAMES),
                epochs=1,
                batch_size=4,
                eval_batch_size=4,
                learning_rate=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                model_dim=16,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
                delta_periods="1,7,90",
                max_train_steps=1,
                seed=17,
                device="cpu",
                overwrite=False,
            )
            runner.train_heads(args)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(summary["best_by_arm"]), set(runner.ARM_NAMES))
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
                self.assertTrue((output_dir / arm / "checkpoint_best_ap.pt").is_file())
                self.assertTrue(
                    (
                        output_dir / arm / "validation_best_ap_predictions.csv"
                    ).is_file()
                )
            status = json.loads(
                (output_dir / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "complete")

    def test_08_stage_a_flattens_batch_times_and_writes_provenance(self) -> None:
        class FakeBackbone(nn.Module):
            embed_dim = 8

            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(0.0))
                self.batch_sizes: list[int] = []

            def forward_features(self, x_dict):
                images = x_dict["imgs"]
                self.batch_sizes.append(int(images.shape[0]))
                pooled = images.mean(dim=(-2, -1))
                padding = torch.zeros(
                    pooled.shape[0],
                    self.embed_dim - pooled.shape[1],
                    device=pooled.device,
                )
                return {"x_norm_clstoken": torch.cat((pooled, padding), dim=1)}

        with tempfile.TemporaryDirectory(prefix="l89-ragged-cache.") as directory:
            root = Path(directory)
            paths = []
            for index in range(9):
                path = root / f"frame-{index}.tif"
                array = np.full((7, 4, 4), 10.0 + index, dtype=np.float32)
                tifffile.imwrite(path, array)
                paths.append(path)
            missing = root / "missing.tif"
            csv_path = root / "train.csv"
            rows = []
            for row in range(3):
                rows.append(
                    {
                        "id": f"id-{row}",
                        "plume_id": f"event-{row}-A",
                        "label": row % 2,
                        "path_t0": str(paths[3 * row]),
                        "path_prev1": str(paths[3 * row + 1]),
                        "path_seasonal": (
                            str(missing) if row == 2 else str(paths[3 * row + 2])
                        ),
                        "t0_image_time": f"2026-01-{10 + row:02d}T00:00:00Z",
                        "prev1_image_time": f"2026-01-{9 + row:02d}T00:00:00Z",
                        "seasonal_image_time": (
                            f"2026-01-{9 + row:02d}T00:00:00Z"
                            if row == 0
                            else f"2025-10-{12 + row:02d}T00:00:00Z"
                        ),
                    }
                )
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            stats_path = root / "stats.json"
            stats_path.write_text(
                json.dumps({"mean": [0.0] * 7, "std": [1.0] * 7}),
                encoding="utf-8",
            )
            weights_path = root / "weights.pt"
            weights_path.write_bytes(b"synthetic-checkpoint")
            output_path = root / "train-cache.pt"
            fake = FakeBackbone()
            args = Namespace(
                csv=str(csv_path),
                split="train",
                output_cache=str(output_path),
                weights=str(weights_path),
                path_columns="path_t0,path_prev1,path_seasonal",
                time_columns=(
                    "t0_image_time,prev1_image_time,seasonal_image_time"
                ),
                label_column="label",
                id_column="id",
                plume_id_column="plume_id",
                event_column="event_group_id",
                band_indices="0,1,2,3,4,5,6",
                stats_json=str(stats_path),
                image_size=4,
                min_valid_fraction=0.75,
                validity_band_index=0,
                zero_invalid_pixels=True,
                batch_size=2,
                num_workers=0,
                prefetch_factor=2,
                persistent_workers=False,
                device="cpu",
                amp_dtype="float32",
                storage_dtype="float32",
                local_cache_dir=str(root / "file-cache"),
                local_cache_bypass_root=str(root),
                local_cache_mode="off",
                local_cache_workers=1,
                local_cache_min_free_gb=0.0,
                max_rows=0,
                row_selection_seed=17,
                max_invalid_t0=0,
                max_read_errors=1,
                log_interval=99,
                overwrite=False,
                debug=False,
            )
            with mock.patch.object(runner, "load_backbone", return_value=fake):
                runner.cache_features(args)
            self.assertEqual(fake.batch_sizes, [6, 3])
            payload = runner.torch_load_trusted(output_path)
            self.assertEqual(tuple(payload["features"].shape), (3, 3, 8))
            self.assertTrue(bool(payload["duplicate_mask"][0, 2]))
            self.assertTrue(bool(payload["unique_mask"][0, 1]))
            self.assertFalse(bool(payload["unique_mask"][0, 2]))
            self.assertFalse(bool(payload["valid_mask"][2, 2]))
            self.assertEqual(payload["delta_days"][0].tolist(), [0.0, -1.0, -1.0])
            self.assertEqual(payload["csv_sha256"], runner.sha256_file(csv_path))
            self.assertEqual(
                payload["weights_sha256"], runner.sha256_file(weights_path)
            )
            self.assertEqual(
                payload["feature_sha256"],
                runner.tensor_sha256(payload["features"]),
            )
            self.assertTrue(
                output_path.with_suffix(output_path.suffix + ".json").is_file()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
