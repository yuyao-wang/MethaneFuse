#!/usr/bin/env python3
"""CPU-only synthetic tests for L89 temporal correspondence pretraining."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_correspondence_pretrain_experiment as experiment,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)


class L89CorrespondencePretrainTests(unittest.TestCase):
    @staticmethod
    def _synthetic_cache(
        split: str,
        *,
        event_prefix: str,
        rows_per_event: int = 2,
        events: int = 10,
    ) -> dict:
        rows = rows_per_event * events
        timepoints = 3
        feature_dim = 8
        generator = torch.Generator().manual_seed(
            101 if split == "train" else 103
        )
        base = torch.randn(rows, feature_dim, generator=generator)
        features = torch.stack(
            (
                base,
                base + 0.05 * torch.randn(
                    rows, feature_dim, generator=generator
                ),
                base + 0.10 * torch.randn(
                    rows, feature_dim, generator=generator
                ),
            ),
            dim=1,
        )
        event_ids = [
            f"{event_prefix}{row // rows_per_event:03d}" for row in range(rows)
        ]
        # Some events are mixed-label to prevent an accidental event-label
        # consistency assumption in the 10% selector.
        labels = torch.tensor(
            [
                (row // rows_per_event + row % rows_per_event) % 2
                for row in range(rows)
            ],
            dtype=torch.long,
        )
        timestamps = torch.tensor(
            [[1_000 + row, 999 + row, 910 + row] for row in range(rows)],
            dtype=torch.int64,
        )
        delta_days = torch.tensor([[0.0, -1.0, -90.0]]).repeat(rows, 1)
        valid = torch.ones(rows, timepoints, dtype=torch.bool)
        duplicate = torch.zeros_like(valid)
        contract = {
            "script_version": cache_runner.SCRIPT_VERSION,
            "csv_sha256": f"{split}-csv",
            "weights_sha256": "a" * 64,
            "input_table_sha256": f"{split}-table",
            "source_rows": rows,
            "selected_rows": rows,
            "row_selection": "all",
            "row_selection_seed": 0,
            "path_columns": ["path_t0", "path_prev1", "path_seasonal"],
            "time_columns": [
                "t0_image_time",
                "prev1_image_time",
                "seasonal_image_time",
            ],
            "role_names": ["t0", "prev1", "seasonal"],
            "band_indices": list(range(7)),
            "normalization_mean": [0.0] * 7,
            "normalization_std": [1.0] * 7,
            "image_size": 4,
            "duplicate_rule": "synthetic",
        }
        return {
            "format_version": cache_runner.CACHE_FORMAT_VERSION,
            "script_version": cache_runner.SCRIPT_VERSION,
            "split": split,
            "features": features,
            "labels": labels,
            "ids": [f"{split}-id-{row:03d}" for row in range(rows)],
            "plume_ids": [f"{event_id}-A" for event_id in event_ids],
            "event_ids": event_ids,
            "event_id_rule": "synthetic",
            "timestamps_utc_ns": timestamps,
            "delta_days": delta_days,
            "role_names": ["t0", "prev1", "seasonal"],
            "role_index": torch.arange(timepoints),
            "t0_index": 0,
            "valid_mask": valid,
            "duplicate_mask": duplicate,
            "unique_mask": valid.clone(),
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
            "csv_sha256": f"{split}-csv",
            "weights_sha256": "a" * 64,
            "feature_sha256": cache_runner.tensor_sha256(features),
        }

    @staticmethod
    def _model_config() -> dict:
        return {
            "feature_dim": 8,
            "num_roles": 3,
            "model_dim": 16,
            "num_heads": 4,
            "depth": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "periods_days": [1.0, 7.0, 90.0],
            "t0_index": 0,
        }

    def test_01_donor_is_deterministic_bijective_and_cross_event(self) -> None:
        events = ["A", "A", "B", "B", "C", "D"]
        first = experiment.build_bijective_cross_event_donors(
            events, seed=37
        )
        second = experiment.build_bijective_cross_event_donors(
            events, seed=37
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(sorted(first.tolist()), list(range(len(events))))
        for target, donor in enumerate(first.tolist()):
            self.assertNotEqual(events[target], events[donor])
        with self.assertRaisesRegex(ValueError, "more than half"):
            experiment.build_bijective_cross_event_donors(
                ["A", "A", "A", "B"], seed=37
            )

    def test_02_balanced_pairs_replace_whole_history_not_t0(self) -> None:
        cache = self._synthetic_cache("train", event_prefix="train-event-")
        indices = cache_runner.select_usable_rows(cache)
        data = cache_runner.take_rows(cache, indices)
        view = experiment.extract_unlabeled_correspondence_view(
            data,
            cache_row_indices=indices,
            t0_index=0,
        )
        pretext = experiment.build_correspondence_pretext(view, seed=19)
        labels = pretext["labels"]
        self.assertEqual(int(labels.sum()), len(labels) // 2)
        self.assertEqual(labels[0::2].tolist(), [1.0] * len(view["event_ids"]))
        self.assertEqual(labels[1::2].tolist(), [0.0] * len(view["event_ids"]))
        donors = pretext["donor_view_indices"]
        self.assertEqual(sorted(donors.tolist()), list(range(len(donors))))
        history = [1, 2]
        for source, donor in enumerate(donors.tolist()):
            positive = 2 * source
            negative = positive + 1
            self.assertTrue(
                torch.equal(
                    pretext["features"][positive, 0],
                    view["features"][source, 0],
                )
            )
            self.assertTrue(
                torch.equal(
                    pretext["features"][negative, 0],
                    view["features"][source, 0],
                )
            )
            for key in ("features", "valid_mask", "unique_mask", "delta_days"):
                self.assertTrue(
                    torch.equal(
                        pretext[key][positive, history],
                        view[key][source, history],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        pretext[key][negative, history],
                        view[key][donor, history],
                    )
                )
        coherent_history = view["features"][:, history].flatten().sort().values
        donor_history = view["features"][donors][:, history].flatten().sort().values
        self.assertTrue(torch.equal(coherent_history, donor_history))

    def test_03_pretext_view_and_examples_do_not_depend_on_methane_labels(
        self,
    ) -> None:
        cache = self._synthetic_cache("train", event_prefix="train-event-")
        indices = cache_runner.select_usable_rows(cache)
        original = cache_runner.take_rows(cache, indices)
        changed = copy.deepcopy(original)
        changed["labels"] = 1.0 - changed["labels"]
        first_view = experiment.extract_unlabeled_correspondence_view(
            original,
            cache_row_indices=indices,
            t0_index=0,
        )
        second_view = experiment.extract_unlabeled_correspondence_view(
            changed,
            cache_row_indices=indices,
            t0_index=0,
        )
        self.assertNotIn("labels", first_view)
        self.assertEqual(
            experiment.correspondence_input_sha256(first_view),
            experiment.correspondence_input_sha256(second_view),
        )
        first = experiment.build_correspondence_pretext(first_view, seed=23)
        second = experiment.build_correspondence_pretext(second_view, seed=23)
        for key in (
            "features",
            "valid_mask",
            "unique_mask",
            "delta_days",
            "labels",
            "donor_view_indices",
        ):
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_04_correspondence_masks_t0_from_attention_context(self) -> None:
        cache = self._synthetic_cache("train", event_prefix="train-event-")
        indices = cache_runner.select_usable_rows(cache)
        data = cache_runner.take_rows(cache, indices)
        view = experiment.extract_unlabeled_correspondence_view(
            data,
            cache_row_indices=indices,
            t0_index=0,
        )
        pretext = experiment.build_correspondence_pretext(view, seed=29)
        model = experiment.build_model(self._model_config())
        attention = model.blocks[0].attention
        with mock.patch.object(
            attention,
            "forward",
            wraps=attention.forward,
        ) as wrapped:
            logits = experiment.correspondence_forward(
                model,
                pretext["features"][:4],
                pretext["valid_mask"][:4],
                pretext["unique_mask"][:4],
                pretext["delta_days"][:4],
                torch.arange(3),
            )
        self.assertEqual(tuple(logits.shape), (4,))
        key_padding_mask = wrapped.call_args.kwargs["key_padding_mask"]
        self.assertTrue(key_padding_mask[:, 0].all())
        self.assertFalse(key_padding_mask[:, 1:].all(dim=1).any())
        no_history = pretext["valid_mask"][:2].clone()
        no_history[:, 1:] = False
        with self.assertRaisesRegex(ValueError, "usable history"):
            experiment.correspondence_forward(
                model,
                pretext["features"][:2],
                no_history,
                no_history,
                pretext["delta_days"][:2],
                torch.arange(3),
            )

    def test_05_transfer_is_exact_and_classifier_is_reset(self) -> None:
        config = self._model_config()
        torch.manual_seed(3)
        source = experiment.build_model(config)
        source_state = {
            key: value.detach().clone() for key, value in source.state_dict().items()
        }
        source_state["classifier.weight"].fill_(99.0)
        source_state["classifier.bias"].fill_(-99.0)
        torch.manual_seed(7)
        target = experiment.build_model(config)
        downstream_initial = {
            key: value.detach().clone() for key, value in target.state_dict().items()
        }
        audit = experiment.transfer_temporal_encoder_state(
            target,
            source_state=source_state,
            downstream_initial_state=downstream_initial,
            source_model_config=config,
            target_model_config=config,
            source_name="synthetic_correspondence",
            allowed_source_only_keys=experiment.CLASSIFIER_STATE_KEYS,
        )
        expected_transfer = sorted(
            set(target.state_dict()) - set(experiment.CLASSIFIER_STATE_KEYS)
        )
        self.assertEqual(audit["transferred_state_keys"], expected_transfer)
        self.assertEqual(
            audit["nontransferred_state_keys"],
            sorted(experiment.CLASSIFIER_STATE_KEYS),
        )
        self.assertEqual(audit["transferred_parameter_count_keys"], 39)
        observed = target.state_dict()
        for key in expected_transfer:
            self.assertTrue(torch.equal(observed[key], source_state[key]))
        for key in experiment.CLASSIFIER_STATE_KEYS:
            self.assertTrue(
                torch.equal(observed[key], downstream_initial[key])
            )
        self.assertEqual(
            audit["classifier_sha256_before"],
            audit["classifier_sha256_after"],
        )

        predictor_source = {
            key: source_state[key] for key in expected_transfer
        }
        predictor_source.update(
            {
                "history_query": torch.zeros(1, 1, 16),
                "null_history_token": torch.zeros(1, 1, 16),
                "output_projection.weight": torch.zeros(8, 16),
                "output_projection.bias": torch.zeros(8),
            }
        )
        external_target = experiment.build_model(config)
        external_audit = experiment.transfer_temporal_encoder_state(
            external_target,
            source_state=predictor_source,
            downstream_initial_state=downstream_initial,
            source_model_config=config,
            target_model_config=config,
            source_name="past_prediction_pretrained",
            allowed_source_only_keys=experiment.PAST_PREDICTOR_ONLY_KEYS,
        )
        self.assertEqual(
            external_audit["ignored_source_only_keys"],
            sorted(experiment.PAST_PREDICTOR_ONLY_KEYS),
        )
        broken = dict(predictor_source)
        del broken["blocks.0.attention.in_proj_weight"]
        with self.assertRaisesRegex(ValueError, "missing temporal"):
            experiment.transfer_temporal_encoder_state(
                external_target,
                source_state=broken,
                downstream_initial_state=downstream_initial,
                source_model_config=config,
                target_model_config=config,
                source_name="broken",
                allowed_source_only_keys=experiment.PAST_PREDICTOR_ONLY_KEYS,
            )

    def test_06_ten_percent_selection_is_event_complete_and_mixed_label_safe(
        self,
    ) -> None:
        event_ids = [
            f"event-{event:02d}" for event in range(10) for _ in range(3)
        ]
        labels = torch.tensor(
            [(event + row) % 2 for event in range(10) for row in range(3)]
        )
        first, first_audit = experiment.select_labeled_event_rows(
            labels,
            event_ids,
            fraction=0.10,
            seed=20_260_727,
        )
        second, second_audit = experiment.select_labeled_event_rows(
            labels,
            event_ids,
            fraction=0.10,
            seed=20_260_727,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_audit, second_audit)
        selected_events = {event_ids[index] for index in first.tolist()}
        self.assertEqual(len(selected_events), 1)
        for event_id in selected_events:
            expected = [
                index
                for index, observed in enumerate(event_ids)
                if observed == event_id
            ]
            actual = [
                index
                for index in first.tolist()
                if event_ids[index] == event_id
            ]
            self.assertEqual(actual, expected)
        self.assertEqual(set(labels[first].tolist()), {0, 1})
        self.assertEqual(first_audit["selected_events"], 1)
        permutation = torch.tensor(
            list(reversed(range(len(event_ids)))), dtype=torch.long
        )
        permuted_events = [event_ids[index] for index in permutation.tolist()]
        _, permuted_audit = experiment.select_labeled_event_rows(
            labels[permutation],
            permuted_events,
            fraction=0.10,
            seed=20_260_727,
        )
        self.assertEqual(
            first_audit["selected_event_sha256"],
            permuted_audit["selected_event_sha256"],
        )

    def test_07_synthetic_end_to_end_has_matched_steps_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="l89-correspondence."
        ) as directory:
            root = Path(directory)
            train_path = root / "train-cache.pt"
            val_path = root / "validation-cache.pt"
            output_dir = root / "run"
            cache_runner.atomic_torch_save(
                train_path,
                self._synthetic_cache(
                    "train", event_prefix="train-event-"
                ),
            )
            cache_runner.atomic_torch_save(
                val_path,
                self._synthetic_cache("val", event_prefix="val-event-"),
            )
            args = Namespace(
                train_cache=str(train_path),
                val_cache=str(val_path),
                output_dir=str(output_dir),
                labeled_event_fractions="0.1,1.0",
                pretrain_epochs=1,
                pretrain_batch_size=8,
                pretrain_learning_rate=1e-3,
                pretrain_weight_decay=0.0,
                max_pretrain_steps=1,
                supervised_epochs=1,
                supervised_batch_size=8,
                eval_batch_size=8,
                supervised_learning_rate=1e-3,
                supervised_weight_decay=0.0,
                max_supervised_steps=1,
                grad_clip=1.0,
                model_dim=16,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
                delta_periods="1,7,90",
                past_predictor_checkpoint=None,
                seed=17,
                device="cpu",
                overwrite=False,
            )
            experiment.run_experiment(args)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["sealed_test_read"])
            self.assertEqual(summary["cache_audit"]["event_overlap"], 0)
            self.assertEqual(
                set(summary["pretraining"]),
                {
                    "correspondence_pretrained",
                    "permuted_pretext_labels_control",
                },
            )
            self.assertEqual(
                summary["pretraining"]["correspondence_pretrained"][
                    "optimizer_steps"
                ],
                summary["pretraining"][
                    "permuted_pretext_labels_control"
                ]["optimizer_steps"],
            )
            self.assertTrue(
                summary["pretraining"]["correspondence_pretrained"][
                    "methane_labels_used"
                ]
                is False
            )
            for fraction_tag in (
                "labeled_events_10pct",
                "labeled_events_100pct",
            ):
                by_arm = summary["results"][fraction_tag]
                self.assertEqual(set(by_arm), set(experiment.BASE_ARMS))
                steps = {
                    record["supervised_optimizer_steps"] for record in by_arm.values()
                }
                rows_seen = {
                    record["supervised_train_rows_seen"]
                    for record in by_arm.values()
                }
                signatures = {
                    record["parameter_signature_sha256"]
                    for record in by_arm.values()
                }
                batch_plans = {
                    record["supervised_batch_plan_sha256"]
                    for record in by_arm.values()
                }
                self.assertEqual(len(steps), 1)
                self.assertEqual(len(rows_seen), 1)
                self.assertEqual(len(signatures), 1)
                self.assertEqual(len(batch_plans), 1)
                for arm, record in by_arm.items():
                    self.assertEqual(
                        record["selection_metric"],
                        "coherent_validation_ap",
                    )
                    self.assertFalse(
                        record["history_shuffled_validation"][
                            "used_for_selection"
                        ]
                    )
                    arm_dir = output_dir / fraction_tag / arm
                    self.assertTrue(
                        (arm_dir / experiment.CHECKPOINT_NAME).is_file()
                    )
                    coherent_path = (
                        arm_dir / experiment.COHERENT_PREDICTION_NAME
                    )
                    shuffled_path = (
                        arm_dir / experiment.SHUFFLED_PREDICTION_NAME
                    )
                    self.assertTrue(coherent_path.is_file())
                    self.assertTrue(shuffled_path.is_file())
                    coherent = pd.read_csv(coherent_path)
                    shuffled = pd.read_csv(shuffled_path)
                    self.assertEqual(
                        coherent["event_id"].tolist(),
                        shuffled["event_id"].tolist(),
                    )
                    self.assertEqual(
                        set(coherent["evaluation_condition"]),
                        {"coherent_history"},
                    )
                    self.assertEqual(
                        set(shuffled["evaluation_condition"]),
                        {"cross_event_shuffled_history"},
                    )
            status = json.loads(
                (output_dir / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "complete")
            self.assertFalse(status["sealed_test_read"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
