#!/usr/bin/env python3
"""CPU-only regression tests for the L89 innovation experiment."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_innovation_pretrain_experiment as innovation,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)


class L89InnovationTests(unittest.TestCase):
    def test_00_serializable_args_excludes_runtime_only_and_callables(self) -> None:
        args = Namespace(
            epochs=3,
            output_dir="/tmp/example",
            resolved_device=torch.device("cpu"),
            func=lambda namespace: namespace,
        )
        self.assertEqual(
            innovation.serializable_args(args),
            {"epochs": 3, "output_dir": "/tmp/example"},
        )

    @staticmethod
    def _synthetic_cache(split: str, *, event_prefix: str) -> dict:
        rows, timepoints, feature_dim = 8, 3, 16
        generator = torch.Generator().manual_seed(31 if split == "train" else 37)
        features = torch.randn(rows, timepoints, feature_dim, generator=generator)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        timestamps = torch.tensor(
            [[1000 + row, 999 + row, 910 + row] for row in range(rows)],
            dtype=torch.int64,
        )
        delta = torch.tensor([[0.0, -1.0, -90.0]]).repeat(rows, 1)
        valid = torch.ones(rows, timepoints, dtype=torch.bool)
        # Exercise valid & unique rather than only the fully dense case.
        valid[0, 2] = False
        unique = valid.clone()
        duplicate = torch.zeros_like(valid)
        contract = {
            "script_version": cache_runner.SCRIPT_VERSION,
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
            "format_version": cache_runner.CACHE_FORMAT_VERSION,
            "script_version": cache_runner.SCRIPT_VERSION,
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
            "valid_fraction": valid.float(),
            "load_status": valid.to(torch.int8),
            "path_columns": ["path_t0", "path_prev1", "path_seasonal"],
            "time_columns": [
                "t0_image_time",
                "prev1_image_time",
                "seasonal_image_time",
            ],
            "input_contract": contract,
            "input_contract_sha256": cache_runner.sha256_bytes(
                cache_runner.canonical_json_bytes(contract)
            ),
            "csv_path": f"/synthetic/{split}.csv",
            "csv_sha256": f"{split}-csv",
            "weights_path": "/synthetic/weights.pt",
            "weights_sha256": "a" * 64,
            "input_table_sha256": f"{split}-table",
            "feature_sha256": cache_runner.tensor_sha256(features),
            "label_sha256": cache_runner.tensor_sha256(labels),
            "timestamp_sha256": cache_runner.tensor_sha256(timestamps),
        }

    def test_01_cross_fit_is_deterministic_and_excludes_complete_events(self) -> None:
        events = ["A", "A", "B", "C", "C", "D", "E", "F", "G", "H"]
        first, audit = innovation.deterministic_event_folds(
            events, num_folds=3, seed=19
        )
        second, _ = innovation.deterministic_event_folds(
            events, num_folds=3, seed=19
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(set(first.tolist()), {0, 1, 2})
        for event in sorted(set(events)):
            assigned = {
                int(first[index])
                for index, value in enumerate(events)
                if value == event
            }
            self.assertEqual(len(assigned), 1)
        for fold in range(3):
            held = {
                events[index]
                for index in range(len(events))
                if int(first[index]) == fold
            }
            fitted = set(events) - held
            self.assertFalse(held & fitted)
            self.assertEqual(audit[fold]["event_overlap"], 0)

    def test_02_predictor_api_is_history_only_and_ignores_t0_changes(self) -> None:
        torch.manual_seed(3)
        rows, timepoints, feature_dim = 4, 3, 8
        base = {
            "features": torch.randn(rows, timepoints, feature_dim),
            "valid_mask": torch.ones(rows, timepoints, dtype=torch.bool),
            "unique_mask": torch.ones(rows, timepoints, dtype=torch.bool),
            "delta_days": torch.tensor([[0.0, -2.0, -90.0]]).repeat(rows, 1),
        }
        changed = {name: value.clone() for name, value in base.items()}
        changed["features"][:, 0] += 10_000.0
        role_index = torch.arange(timepoints)
        history_a = innovation.extract_history_only_inputs(
            base, role_index=role_index, t0_index=0
        )
        history_b = innovation.extract_history_only_inputs(
            changed, role_index=role_index, t0_index=0
        )
        for name in (
            "features",
            "valid_mask",
            "unique_mask",
            "delta_days",
            "role_index",
        ):
            self.assertTrue(torch.equal(history_a[name], history_b[name]))
        self.assertFalse(torch.equal(history_a["target_t0"], history_b["target_t0"]))

        model = innovation.HistoryOnlyPredictor(
            feature_dim=feature_dim,
            num_roles=timepoints,
            t0_index=0,
            model_dim=16,
            num_heads=4,
            dropout=0.0,
        ).eval()
        with torch.inference_mode():
            prediction_a = model(
                history_a["features"],
                history_a["valid_mask"],
                history_a["unique_mask"],
                history_a["role_index"],
                history_a["delta_days"],
            )
            prediction_b = model(
                history_b["features"],
                history_b["valid_mask"],
                history_b["unique_mask"],
                history_b["role_index"],
                history_b["delta_days"],
            )
        self.assertTrue(torch.equal(prediction_a, prediction_b))
        with self.assertRaisesRegex(ValueError, "t0 role"):
            model(
                history_a["features"],
                history_a["valid_mask"],
                history_a["unique_mask"],
                torch.tensor([0, 2]),
                history_a["delta_days"],
            )
        future_delta = history_a["delta_days"].clone()
        future_delta[:, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "at or after t0"):
            model(
                history_a["features"],
                history_a["valid_mask"],
                history_a["unique_mask"],
                history_a["role_index"],
                future_delta,
            )

    def test_03_innovation_is_manual_residual_with_stopped_background(self) -> None:
        z0 = torch.tensor([[3.0, -2.0], [1.0, 4.0]], requires_grad=True)
        background = torch.tensor(
            [[1.0, 5.0], [-2.0, 0.5]], requires_grad=True
        )
        residual = innovation.compute_innovation(z0, background)
        expected = torch.tensor([[2.0, -7.0], [3.0, 3.5]])
        self.assertTrue(torch.equal(residual, expected))
        residual.sum().backward()
        self.assertTrue(torch.equal(z0.grad, torch.ones_like(z0)))
        self.assertIsNone(background.grad)

    def test_04_history_shuffle_moves_whole_history_across_events(self) -> None:
        events = ["A", "A", "B", "C", "D", "D"]
        donors = innovation.build_history_shuffle_donors(events, seed=23)
        rows = len(events)
        history = {
            "features": torch.arange(rows * 2 * 3).reshape(rows, 2, 3).float(),
            "valid_mask": torch.tensor(
                [[True, False], [True, True], [False, True]] * 2
            ),
            "unique_mask": torch.tensor(
                [[True, False], [True, False], [False, True]] * 2
            ),
            "delta_days": torch.arange(rows * 2).reshape(rows, 2).float(),
            "role_index": torch.tensor([1, 2]),
        }
        shuffled = innovation.apply_history_only_donors(history, donors)
        for row, donor in enumerate(donors.tolist()):
            self.assertNotEqual(events[row], events[donor])
        for name in ("features", "valid_mask", "unique_mask", "delta_days"):
            self.assertTrue(torch.equal(shuffled[name], history[name][donors]))
        self.assertTrue(
            torch.equal(shuffled["role_index"], history["role_index"])
        )

    def test_05_five_arms_have_matched_shapes_and_parameter_signatures(self) -> None:
        torch.manual_seed(5)
        z0 = torch.randn(6, 8)
        background = torch.randn(6, 8)
        shuffled = torch.randn(6, 8)
        slot_shapes = []
        signatures = []
        state_shapes = []
        for arm in innovation.DOWNSTREAM_ARM_NAMES:
            slots = innovation.build_downstream_slots(
                arm=arm,
                target_t0=z0,
                predicted_background=background,
                shuffled_predicted_background=shuffled,
            )
            slot_shapes.append(tuple(slots.shape))
            model = innovation.MatchedTwoSlotClassifier(
                feature_dim=8, model_dim=16, dropout=0.0
            )
            signatures.append(cache_runner.model_parameter_signature(model))
            state_shapes.append(
                {
                    name: tuple(value.shape)
                    for name, value in model.state_dict().items()
                }
            )
        self.assertEqual(set(slot_shapes), {(6, 2, 8)})
        for signature in signatures[1:]:
            self.assertEqual(signature, signatures[0])
        for shapes in state_shapes[1:]:
            self.assertEqual(shapes, state_shapes[0])
        plus = innovation.build_downstream_slots(
            arm="t0_plus_innovation",
            target_t0=z0,
            predicted_background=background,
            shuffled_predicted_background=shuffled,
        )
        recovered_background = plus[:, 0] - plus[:, 1]
        self.assertTrue(torch.allclose(recovered_background, background))

    def test_06_predictor_and_all_heads_forward_backward_are_finite(self) -> None:
        torch.manual_seed(7)
        rows, history_length, feature_dim = 6, 2, 8
        predictor = innovation.HistoryOnlyPredictor(
            feature_dim=feature_dim,
            num_roles=3,
            t0_index=0,
            model_dim=16,
            num_heads=4,
            dropout=0.0,
        )
        histories = torch.randn(rows, history_length, feature_dim)
        valid = torch.zeros(rows, history_length, dtype=torch.bool)
        unique = torch.zeros_like(valid)
        delta = torch.tensor([[-1.0, -90.0]]).repeat(rows, 1)
        target = torch.randn(rows, feature_dim)
        prediction = predictor(
            histories, valid, unique, torch.tensor([1, 2]), delta
        )
        objective, _, _ = innovation.predictor_loss_terms(
            prediction, target, cosine_weight=0.1
        )
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        self.assertIsNotNone(predictor.output_projection.weight.grad)

        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        for arm in innovation.DOWNSTREAM_ARM_NAMES:
            slots = innovation.build_downstream_slots(
                arm=arm,
                target_t0=target,
                predicted_background=prediction.detach(),
                shuffled_predicted_background=torch.roll(
                    prediction.detach(), shifts=1, dims=0
                ),
            )
            model = innovation.MatchedTwoSlotClassifier(
                feature_dim=feature_dim, model_dim=16, dropout=0.0
            )
            logits = model(slots)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.mlp[-1].weight.grad)

    def test_07_end_to_end_cross_fit_to_five_downstream_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-innovation.") as directory:
            root = Path(directory)
            train_path = root / "train.pt"
            val_path = root / "val.pt"
            pretrain_dir = root / "pretrain"
            downstream_dir = root / "downstream"
            cache_runner.atomic_torch_save(
                train_path,
                self._synthetic_cache("train", event_prefix="train-event-"),
            )
            cache_runner.atomic_torch_save(
                val_path,
                self._synthetic_cache("val", event_prefix="val-event-"),
            )
            pretrain_args = Namespace(
                train_cache=str(train_path),
                val_cache=str(val_path),
                output_dir=str(pretrain_dir),
                folds=2,
                epochs=1,
                batch_size=4,
                eval_batch_size=4,
                learning_rate=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                cosine_weight=0.1,
                model_dim=16,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
                delta_periods="1,7,90",
                max_train_steps=1,
                seed=29,
                device="cpu",
                overwrite=False,
            )
            innovation.run_pretrain(pretrain_args)
            pretrain_summary = json.loads(
                (pretrain_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                pretrain_summary["cross_fitting"]["fold_event_overlap"], 0
            )
            self.assertTrue(
                pretrain_summary["cross_fitting"]["oof_coverage_exactly_once"]
            )
            self.assertFalse(pretrain_summary["sealed_test_read"])
            for fold in pretrain_summary["cross_fitting"]["fold_results"]:
                self.assertEqual(fold["event_overlap"], 0)
                self.assertIn("mse", fold["held_out_metrics"])
                self.assertEqual(
                    fold["checkpoint_sha256"],
                    cache_runner.sha256_file(Path(fold["checkpoint"])),
                )
            oof = cache_runner.torch_load_trusted(
                pretrain_dir / "oof_train_predictions.pt"
            )
            self.assertEqual(set(oof["fold_index"].tolist()), {0, 1})
            self.assertEqual(
                tuple(oof["predicted_background"].shape), (8, 16)
            )
            for row, donor in enumerate(oof["shuffle_donor_row"].tolist()):
                self.assertNotEqual(
                    oof["event_ids"][row], oof["event_ids"][donor]
                )
            self.assertEqual(
                oof["tensor_sha256"],
                innovation.prediction_tensor_hashes(oof),
            )

            downstream_args = Namespace(
                train_cache=str(train_path),
                val_cache=str(val_path),
                pretrain_dir=str(pretrain_dir),
                output_dir=str(downstream_dir),
                arms=",".join(innovation.DOWNSTREAM_ARM_NAMES),
                epochs=1,
                batch_size=4,
                eval_batch_size=4,
                learning_rate=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                model_dim=16,
                dropout=0.0,
                max_train_steps=1,
                seed=31,
                device="cpu",
                overwrite=False,
            )
            innovation.run_downstream(downstream_args)
            downstream_summary = json.loads(
                (downstream_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(downstream_summary["best_by_arm"]),
                set(innovation.DOWNSTREAM_ARM_NAMES),
            )
            self.assertFalse(downstream_summary["sealed_test_read"])
            self.assertIn(
                "redundant",
                downstream_summary["representation_contract"][
                    "t0_plus_innovation_interpretation"
                ],
            )
            for arm in innovation.DOWNSTREAM_ARM_NAMES:
                metrics = downstream_summary["best_by_arm"][arm]["validation"]
                for key in (
                    "ap",
                    "auc",
                    "macro_f1_at_0_5",
                    "balanced_accuracy_at_0_5",
                    "pred_positive_rate_at_0_5",
                ):
                    self.assertIn(key, metrics)
                self.assertTrue(
                    (downstream_dir / arm / "checkpoint_best_ap.pt").is_file()
                )
                self.assertTrue(
                    (
                        downstream_dir
                        / arm
                        / "validation_best_ap_predictions.csv"
                    ).is_file()
                )
            status = json.loads(
                (downstream_dir / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
