#!/usr/bin/env python3
"""CPU contracts for the train/dev-only TEMPO L89 patch pipeline."""

from __future__ import annotations

import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tempo_l89_patch as tempo  # noqa: E402
import audit_tempo_patch_history_shuffle as shuffle_audit  # noqa: E402


class NestedProjectionTests(unittest.TestCase):
    def test_nested_extension_preserves_base_prefix_bit_exactly(self):
        base = tempo.deterministic_orthogonal_projection(
            16, 4, seed=11
        )
        extended = tempo.nested_extend_orthogonal_projection(
            16,
            8,
            base_dim=4,
            base_seed=11,
            extension_seed=29,
        )
        self.assertTrue(torch.equal(extended[:, :4], base))

    def test_nested_extension_is_orthogonal_and_metadata_audited(self):
        extended = tempo.nested_extend_orthogonal_projection(
            16,
            8,
            base_dim=4,
            base_seed=11,
            extension_seed=29,
        )
        error = (
            extended.T @ extended - torch.eye(8)
        ).abs().max()
        self.assertLess(float(error), 1e-5)
        metadata = tempo.nested_projection_metadata(
            extended,
            base_dim=4,
            base_seed=11,
            extension_seed=29,
        )
        self.assertTrue(metadata["first_base_columns_array_equal"])
        self.assertEqual(metadata["first_base_columns_max_abs_error"], 0.0)
        self.assertLess(metadata["max_abs_qtq_minus_i"], 1e-5)

    def test_extension_seed_changes_only_extra_columns(self):
        first = tempo.nested_extend_orthogonal_projection(
            16,
            8,
            base_dim=4,
            base_seed=11,
            extension_seed=29,
        )
        second = tempo.nested_extend_orthogonal_projection(
            16,
            8,
            base_dim=4,
            base_seed=11,
            extension_seed=31,
        )
        self.assertTrue(torch.equal(first[:, :4], second[:, :4]))
        self.assertFalse(torch.equal(first[:, 4:], second[:, 4:]))


class SpatialMatchingTests(unittest.TestCase):
    @staticmethod
    def _single_match(
        *,
        radius: int,
        query_coordinate: tuple[int, int],
        source_coordinate: tuple[int, int],
    ) -> float:
        height = width = 5
        patches = height * width
        query = torch.tensor([[[1.0, 0.0]]]).expand(1, patches, 2).clone()
        keys = torch.tensor([[[[-1.0, 0.0]]]]).expand(
            1, 1, patches, 2
        ).clone()
        values = torch.arange(patches, dtype=torch.float32).reshape(
            1, 1, patches, 1
        )
        source = source_coordinate[0] * width + source_coordinate[1]
        target = query_coordinate[0] * width + query_coordinate[1]
        keys[0, 0, source] = torch.tensor([1.0, 0.0])
        aligned, entropy = tempo.local_soft_match(
            query,
            keys,
            values,
            grid_shape=(height, width),
            radius=radius,
            temperature=0.01,
        )
        if not torch.isfinite(entropy).all():
            raise AssertionError("matcher entropy is non-finite")
        return float(aligned[0, 0, target, 0])

    def test_3x3_neighbourhood_retrieves_one_patch_shift(self):
        observed = self._single_match(
            radius=1,
            query_coordinate=(2, 2),
            source_coordinate=(2, 3),
        )
        self.assertAlmostEqual(observed, 13.0, places=4)

    def test_radius_zero_is_exact_same_pixel_without_search(self):
        batch, histories, height, width, rank, value_dim = 2, 3, 3, 4, 5, 2
        patches = height * width
        query = torch.randn(batch, patches, rank)
        keys = torch.randn(batch, histories, patches, rank)
        values = torch.randn(batch, histories, patches, value_dim)
        aligned, entropy = tempo.local_soft_match(
            query,
            keys,
            values,
            grid_shape=(height, width),
            radius=0,
            temperature=0.1,
        )
        torch.testing.assert_close(aligned, values.float(), atol=0, rtol=0)
        torch.testing.assert_close(
            entropy,
            torch.zeros_like(entropy),
            atol=0,
            rtol=0,
        )

    def test_radius_zero_cannot_retrieve_adjacent_patch(self):
        observed = self._single_match(
            radius=0,
            query_coordinate=(2, 2),
            source_coordinate=(2, 3),
        )
        self.assertAlmostEqual(observed, 12.0, places=4)

    def test_5x5_neighbourhood_retrieves_two_patch_shift(self):
        observed = self._single_match(
            radius=2,
            query_coordinate=(2, 2),
            source_coordinate=(2, 4),
        )
        self.assertAlmostEqual(observed, 14.0, places=4)

    def test_3x3_cannot_retrieve_two_patch_shift(self):
        observed = self._single_match(
            radius=1,
            query_coordinate=(2, 2),
            source_coordinate=(2, 4),
        )
        self.assertNotAlmostEqual(observed, 14.0, places=2)

    def test_border_neighbourhood_never_wraps(self):
        # Source (4,4) must not become a neighbour of query (0,0).
        observed = self._single_match(
            radius=2,
            query_coordinate=(0, 0),
            source_coordinate=(4, 4),
        )
        self.assertLess(observed, 13.0)


class OnsetOperatorTests(unittest.TestCase):
    def test_history_pairwise_normality_is_explicitly_subtracted(self):
        current = torch.tensor([[[4.0]]])
        stable_history = torch.tensor([[[[1.0]], [[1.0]]]])
        weights = torch.tensor([[0.5, 0.5]])
        valid = torch.tensor([[True, True]])
        stable = tempo.temporal_onset_components(
            current,
            stable_history,
            weights,
            valid,
            normality_scale=1.0,
        )
        torch.testing.assert_close(
            stable["current_absolute"], torch.tensor([[[3.0]]])
        )
        torch.testing.assert_close(
            stable["history_normality"], torch.tensor([[[0.0]]])
        )
        torch.testing.assert_close(stable["excess"], torch.tensor([[[3.0]]]))

        varying_history = torch.tensor([[[[1.0]], [[5.0]]]])
        varying = tempo.temporal_onset_components(
            current,
            varying_history,
            weights,
            valid,
            normality_scale=1.0,
        )
        torch.testing.assert_close(
            varying["history_normality"], torch.tensor([[[4.0]]])
        )
        torch.testing.assert_close(varying["excess"], torch.tensor([[[0.0]]]))

    def test_no_normality_mode_zeroes_all_normality_derived_channels(self):
        shared = {
            "signed_change": torch.tensor([[[0.2, -0.1]]]),
            "current_absolute": torch.tensor([[[1.0, 2.0]]]),
        }
        low = {
            **shared,
            "history_normality": torch.zeros(1, 1, 2),
            "excess": torch.tensor([[[1.0, 2.0]]]),
            "ratio": torch.tensor([[[0.5, 0.7]]]),
        }
        high = {
            **shared,
            "history_normality": torch.tensor([[[9.0, 4.0]]]),
            "excess": torch.zeros(1, 1, 2),
            "ratio": torch.tensor([[[0.1, 0.2]]]),
        }
        low_a0, low_effective = tempo.assemble_onset_features(
            low, use_normality_features=False
        )
        high_a0, high_effective = tempo.assemble_onset_features(
            high, use_normality_features=False
        )
        self.assertTrue(torch.equal(low_a0, high_a0))
        self.assertEqual(
            int(torch.count_nonzero(low_effective["history_normality"])), 0
        )
        self.assertEqual(int(torch.count_nonzero(low_effective["ratio"])), 0)
        torch.testing.assert_close(
            low_effective["excess"], shared["current_absolute"]
        )
        low_a1, _ = tempo.assemble_onset_features(
            low, use_normality_features=True
        )
        high_a1, _ = tempo.assemble_onset_features(
            high, use_normality_features=True
        )
        self.assertFalse(torch.equal(low_a1, high_a1))
        # A matched readout must also be identical for A0 inputs.
        readout = torch.nn.Linear(low_a0.shape[-1], 1)
        torch.testing.assert_close(readout(low_a0), readout(high_a0))

    def test_no_history_produces_exact_fallback_even_after_gate_opens(self):
        model = tempo.TempoPatchHead(
            8,
            match_rank=4,
            value_dim=4,
            hidden_dim=8,
            radius=1,
        )
        with torch.no_grad():
            model.residual_gate.fill_(1.0)
        base = torch.tensor([-1.25, 2.75])
        tokens = torch.randn(2, 3, 9, 8)
        valid = torch.tensor(
            [[True, False, False], [True, False, False]]
        )
        delta = torch.tensor(
            [[0.0, float("nan"), float("nan")]] * 2
        )
        quality = torch.zeros(2, 3)
        output = model(
            base,
            tokens,
            valid,
            delta,
            quality,
            t0_index=0,
            grid_shape=(3, 3),
        )
        self.assertTrue(torch.equal(output.logits, base))
        self.assertEqual(int(torch.count_nonzero(output.patch_scores)), 0)
        self.assertEqual(int(torch.count_nonzero(output.history_weights)), 0)


class TrainingContractTests(unittest.TestCase):
    def test_zero_init_is_bit_exact_and_receives_classification_gradient(self):
        model = tempo.TempoPatchHead(
            8,
            match_rank=4,
            value_dim=4,
            hidden_dim=8,
            radius=1,
        )
        base = torch.tensor([-0.2, 0.4])
        tokens = torch.randn(2, 3, 9, 8)
        valid = torch.ones(2, 3, dtype=torch.bool)
        delta = torch.tensor([[0.0, -30.0, -365.0]] * 2)
        quality = torch.ones(2, 3)
        output = model(
            base,
            tokens,
            valid,
            delta,
            quality,
            t0_index=0,
            grid_shape=(3, 3),
        )
        self.assertTrue(model.exact_noop)
        self.assertTrue(torch.equal(output.logits, base))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits, torch.tensor([0.0, 1.0])
        )
        loss.backward()
        self.assertIsNotNone(model.residual_gate.grad)
        self.assertGreater(abs(float(model.residual_gate.grad)), 0.0)

    def test_readout_zero_is_bit_exact_and_removes_random_gate_sign(self):
        model = tempo.TempoPatchHead(
            8,
            match_rank=4,
            value_dim=4,
            hidden_dim=8,
            radius=1,
            zero_init_mode="readout",
        )
        base = torch.tensor([-0.2, 0.4])
        tokens = torch.randn(2, 3, 9, 8)
        valid = torch.ones(2, 3, dtype=torch.bool)
        delta = torch.tensor([[0.0, -30.0, -365.0]] * 2)
        quality = torch.ones(2, 3)
        output = model(
            base,
            tokens,
            valid,
            delta,
            quality,
            t0_index=0,
            grid_shape=(3, 3),
        )
        self.assertTrue(model.exact_noop)
        self.assertTrue(torch.equal(output.logits, base))
        self.assertEqual(float(model.residual_gate), 1.0)
        self.assertFalse(model.residual_gate.requires_grad)
        self.assertEqual(
            int(torch.count_nonzero(model.patch_readout.weight)), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(model.patch_readout.bias)), 0
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits, torch.tensor([0.0, 1.0])
        )
        loss.backward()
        self.assertIsNone(model.residual_gate.grad)
        self.assertIsNotNone(model.patch_readout.weight.grad)
        self.assertGreater(
            float(model.patch_readout.weight.grad.abs().sum()), 0.0
        )

    def test_all_negative_event_mask_uses_complete_event(self):
        labels = torch.tensor([0, 0, 0, 1, 0])
        events = ["a", "a", "mixed", "mixed", "b"]
        observed = tempo.all_negative_event_mask(labels, events)
        self.assertEqual(
            observed.tolist(), [True, True, False, False, True]
        )

    def test_null_loss_has_gradient_and_empty_case_is_zero(self):
        scores = torch.tensor(
            [[2.0, -1.0, 0.5], [-2.0, -3.0, -4.0]],
            requires_grad=True,
        )
        loss = tempo.negative_event_topk_null_loss(
            scores,
            torch.tensor([True, False]),
            topk_fraction=1 / 3,
        )
        loss.backward()
        self.assertGreater(float(scores.grad[0].abs().sum()), 0.0)
        empty = tempo.negative_event_topk_null_loss(
            scores.detach(),
            torch.tensor([False, False]),
            topk_fraction=1 / 3,
        )
        self.assertEqual(float(empty), 0.0)


class HistoryShuffleContractTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.arange(
            6 * 3 * 2 * 2, dtype=torch.float32
        ).reshape(6, 3, 2, 2)
        self.events = ["a", "a", "b", "b", "c", "c"]
        self.donors = shuffle_audit.l89.build_cross_event_donor_indices(
            self.events, seed=19
        )

    def test_cross_event_donor_map_has_no_self_event_pair(self):
        for target, donor in enumerate(self.donors.tolist()):
            self.assertNotEqual(self.events[target], self.events[donor])

    def test_patch_shuffle_preserves_t0_exactly(self):
        shuffled = shuffle_audit.apply_patch_history_donors(
            self.tokens, self.donors, t0_index=0
        )
        torch.testing.assert_close(
            shuffled[:, 0], self.tokens[:, 0], atol=0, rtol=0
        )

    def test_patch_shuffle_replaces_only_history_with_donor_content(self):
        original = self.tokens.clone()
        shuffled = shuffle_audit.apply_patch_history_donors(
            self.tokens, self.donors, t0_index=0
        )
        torch.testing.assert_close(
            shuffled[:, 1:],
            self.tokens[self.donors, 1:],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            self.tokens, original, atol=0, rtol=0
        )

    def test_nonpatch_context_tensors_remain_bit_exact(self):
        base = torch.randn(6)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        valid = torch.randint(0, 2, (6, 3), dtype=torch.bool)
        delta = torch.randn(6, 3)
        quality = torch.rand(6, 3)
        snapshots = [
            value.clone() for value in (base, labels, valid, delta, quality)
        ]
        shuffle_audit.apply_patch_history_donors(
            self.tokens, self.donors, t0_index=0
        )
        for value, expected in zip(
            (base, labels, valid, delta, quality), snapshots
        ):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)

    def test_mask_matched_donors_are_cross_event_and_zero_mismatch(self):
        mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
            ],
            dtype=torch.bool,
        )
        events = ["a", "b", "a", "b", "c", "c"]
        donors, eligible, receipt = (
            shuffle_audit.build_unique_mask_matched_cross_event_donors(
                mask, events, t0_index=0, seed=23
            )
        )
        self.assertTrue(bool(eligible.all()))
        self.assertEqual(receipt["excluded_rows"], 0)
        self.assertEqual(
            receipt["donor_full_mask_target_mask_mismatch_slots"], 0
        )
        self.assertEqual(
            receipt["target_valid_to_donor_invalid_history_slots"], 0
        )
        for target, donor in enumerate(donors.tolist()):
            self.assertNotEqual(events[target], events[donor])
            torch.testing.assert_close(
                mask[target], mask[donor], atol=0, rtol=0
            )

    def test_unmatchable_mask_group_is_excluded_and_left_unchanged(self):
        mask = torch.tensor(
            [[1, 1, 1], [1, 1, 1], [1, 0, 1]],
            dtype=torch.bool,
        )
        events = ["a", "b", "solo"]
        donors, eligible, receipt = (
            shuffle_audit.build_unique_mask_matched_cross_event_donors(
                mask, events, t0_index=0, seed=23
            )
        )
        self.assertEqual(eligible.tolist(), [True, True, False])
        self.assertEqual(donors[2].item(), 2)
        self.assertEqual(receipt["excluded_row_indices"], [2])
        shuffled = shuffle_audit.apply_patch_history_donors(
            self.tokens[:3], donors, t0_index=0
        )
        torch.testing.assert_close(
            shuffled[2], self.tokens[2], atol=0, rtol=0
        )

    def test_cpu_dry_run(self):
        result = tempo.run_synthetic_dry_run(seed=7)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["epoch_zero_exact_base_logit"])
        self.assertGreater(result["finite_positive_gradient_sum"], 0.0)


class PathAndShardContractTests(unittest.TestCase):
    def test_held_out_path_tokens_are_rejected(self):
        tempo.assert_development_path(
            Path("/tmp/tempo_dev/cache"), purpose="development"
        )
        for token in ("test", "sealed", "holdout"):
            with self.assertRaisesRegex(ValueError, "held-out"):
                tempo.assert_development_path(
                    Path(f"/tmp/tempo_{token}/cache"), purpose="forbidden"
                )

    @staticmethod
    def _write_cache(
        root: Path,
        *,
        split: str,
        event_ids: list[str],
    ) -> Path:
        cache_dir = root / f"{split}_cache"
        cache_dir.mkdir(parents=True)
        projection = tempo.projection_metadata(
            tempo.deterministic_orthogonal_projection(8, 4, seed=17),
            seed=17,
        )
        configuration = {
            "split": split,
            "weights_sha256": "weights",
            "base_head_sha256": "base",
            "projection": projection,
            "timepoints": 3,
            "t0_index": 0,
            "input_contract_sha256": "contract",
            "comparable_input_contract_sha256": "comparable-contract",
            "projection_storage_dtype": "torch.float16",
            "amp_dtype": "float16",
            "raw_768d_patch_tokens_persisted": False,
        }
        configuration_sha = tempo.canonical_json_sha256(configuration)
        rows = len(event_ids)
        patches = torch.arange(
            rows * 3 * 4 * 4, dtype=torch.float16
        ).reshape(rows, 3, 4, 4)
        labels = torch.tensor(
            [index % 2 for index in range(rows)], dtype=torch.long
        )
        shard = {
            "format_version": tempo.SHARD_VERSION,
            "configuration_sha256": configuration_sha,
            "row_indices": torch.arange(rows, dtype=torch.long),
            "patch_tokens": patches,
            "base_logits": torch.linspace(-0.4, 0.4, rows),
            "labels": labels,
            "unique_mask": torch.ones(rows, 3, dtype=torch.bool),
            "delta_days": torch.tensor(
                [[0.0, -30.0, -365.0]] * rows
            ),
            "quality": torch.ones(rows, 3, dtype=torch.float16),
            "ids": [f"id-{index}" for index in range(rows)],
            "event_ids": event_ids,
            "raw_768d_patch_tokens_persisted": False,
            "tensor_sha256": {"patch_tokens": tempo.tensor_sha256(patches)},
        }
        shard_path = cache_dir / "shard_00000.pt"
        tempo.atomic_torch(shard_path, shard)
        record = {
            "shard_index": 0,
            "row_start": 0,
            "row_stop": rows,
            "rows": rows,
            "file": shard_path.name,
            "sha256": tempo.sha256_file(shard_path),
            "patch_tokens_sha256": tempo.tensor_sha256(patches),
            "shape": list(patches.shape),
        }
        identity = {
            "ids": [f"id-{index}" for index in range(rows)],
            "plume_ids": [f"plume-{index}" for index in range(rows)],
            "event_ids": event_ids,
            "labels": labels.tolist(),
        }
        identity["identity_sha256"] = tempo.canonical_json_sha256(
            {
                "ids": identity["ids"],
                "event_ids": identity["event_ids"],
                "labels": identity["labels"],
            }
        )
        manifest = {
            "format_version": tempo.CACHE_VERSION,
            "status": "complete",
            "configuration": configuration,
            "configuration_sha256": configuration_sha,
            "grid_shape": [2, 2],
            "rows": rows,
            "shards": [record],
            "identity": identity,
        }
        manifest["manifest_content_sha256"] = tempo.canonical_json_sha256(
            {
                "configuration_sha256": configuration_sha,
                "shards": manifest["shards"],
                "identity_sha256": identity["identity_sha256"],
            }
        )
        manifest_path = cache_dir / "manifest.json"
        tempo.atomic_json(manifest_path, manifest)
        return manifest_path

    def test_valid_shard_is_resumable_and_sha_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="tempo_contract_") as raw:
            root = Path(raw)
            manifest_path = self._write_cache(
                root, split="train", event_ids=["a", "b"]
            )
            loaded = tempo.load_manifest(
                manifest_path, expected_split="train"
            )
            self.assertEqual(loaded["status"], "complete")
            shard_path = manifest_path.parent / "shard_00000.pt"
            with shard_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "file SHA mismatch"):
                tempo.load_manifest(manifest_path, expected_split="train")

    def test_event_overlap_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory(prefix="tempo_contract_") as raw:
            root = Path(raw)
            train = self._write_cache(
                root, split="train", event_ids=["shared", "train-only"]
            )
            val = self._write_cache(
                root, split="val", event_ids=["shared", "val-only"]
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                tempo.validate_cache_pair(train, val)

    def test_manifest_configuration_sha_mismatch_blocks_resume(self):
        with tempfile.TemporaryDirectory(prefix="tempo_contract_") as raw:
            root = Path(raw)
            manifest_path = self._write_cache(
                root, split="train", event_ids=["a", "b"]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["configuration_sha256"] = "different"
            tempo.atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "configuration SHA mismatch"):
                tempo.load_manifest(manifest_path, expected_split="train")

    @staticmethod
    def _write_prediction_csv(
        path: Path,
        manifest_path: Path,
    ) -> Path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = int(manifest["rows"])
        frame = {
            "id": manifest["identity"]["ids"],
            "event_id": manifest["identity"]["event_ids"],
            "label": manifest["identity"]["labels"],
            "probability": [
                0.15 + 0.7 * index / max(1, rows - 1)
                for index in range(rows)
            ],
        }
        import pandas as pd

        pd.DataFrame(frame).to_csv(path, index=False)
        return path

    def test_prediction_overlay_roundtrip_and_family_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="tempo_contract_") as raw:
            root = Path(raw)
            train_manifest = self._write_cache(
                root, split="train", event_ids=["tr-a", "tr-b"]
            )
            val_manifest = self._write_cache(
                root, split="val", event_ids=["va-a", "va-b"]
            )
            train_csv = self._write_prediction_csv(
                root / "train_predictions.csv", train_manifest
            )
            val_csv = self._write_prediction_csv(
                root / "val_predictions.csv", val_manifest
            )
            train_overlay = root / "train_overlay.pt"
            val_overlay = root / "val_overlay.pt"
            common = {
                "feature_cache": "",
                "head_checkpoint": "",
                "head_config_json": "",
                "verify_predictions_csv": "",
                "probability_column": "probability",
                "probability_clip": 1e-7,
                "source_model_sha256": "model-sha",
                "batch_size": 16,
            }
            tempo.command_rebase(
                argparse.Namespace(
                    split="train",
                    patch_manifest=str(train_manifest),
                    family="event-p0",
                    output_overlay=str(train_overlay),
                    predictions_csv=str(train_csv),
                    **common,
                )
            )
            tempo.command_rebase(
                argparse.Namespace(
                    split="val",
                    patch_manifest=str(val_manifest),
                    family="event-p5",
                    output_overlay=str(val_overlay),
                    predictions_csv=str(val_csv),
                    **common,
                )
            )
            train_loaded = tempo.load_manifest(
                train_manifest, expected_split="train"
            )
            logits, audit = tempo.load_base_overlay(
                train_overlay, train_loaded, expected_split="train"
            )
            expected = torch.tensor([0.15, 0.85])
            torch.testing.assert_close(
                torch.sigmoid(logits), expected, atol=5e-7, rtol=0
            )
            self.assertEqual(audit["family"], "event-p0")
            with self.assertRaisesRegex(ValueError, "family mismatch"):
                tempo.command_train(
                    argparse.Namespace(
                        train_manifest=str(train_manifest),
                        val_manifest=str(val_manifest),
                        train_base_overlay=str(train_overlay),
                        val_base_overlay=str(val_overlay),
                        output_dir=str(root / "development_mismatch"),
                        resume=False,
                        seed=1,
                    )
                )

    def test_one_epoch_streamed_train_command(self):
        with tempfile.TemporaryDirectory(prefix="tempo_contract_") as raw:
            root = Path(raw)
            train = self._write_cache(
                root,
                split="train",
                event_ids=[f"tr-{index}" for index in range(8)],
            )
            val = self._write_cache(
                root,
                split="val",
                event_ids=[f"va-{index}" for index in range(6)],
            )
            output = root / "development_result"
            tempo.command_train(
                argparse.Namespace(
                    train_manifest=str(train),
                    val_manifest=str(val),
                    train_base_overlay="",
                    val_base_overlay="",
                    output_dir=str(output),
                    device="cpu",
                    match_rank=2,
                    value_dim=2,
                    hidden_dim=4,
                    radius=1,
                    temperature=0.1,
                    topk_fraction=0.25,
                    normality_scale=1.0,
                    use_normality_features=True,
                    residual_cap=1.5,
                    null_weight=0.1,
                    null_margin=0.0,
                    epochs=1,
                    batch_size=4,
                    eval_batch_size=3,
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                    grad_clip=1.0,
                    patience=1,
                    min_delta=0.0,
                    max_steps_per_epoch=0,
                    selection_metric="event_balanced_ap",
                    resume=False,
                    seed=23,
                )
            )
            result = json.loads(
                (output / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue(
                result["matching_contract"]["epoch_zero_exact_base_logit"]
            )
            zero = result["epoch_zero"]
            self.assertLessEqual(
                zero["probability_max_abs_vs_frozen_base"],
                zero["probability_tolerance"],
            )
            self.assertEqual(zero["probability_tolerance"], 2e-7)
            zero_metrics = zero["validation"]
            self.assertIn(
                "event_balanced_macro_f1_selected", zero_metrics
            )
            self.assertIn(
                "event_balanced_positive_f1_selected", zero_metrics
            )
            self.assertAlmostEqual(
                zero_metrics["event_balanced_macro_f1_selected"],
                zero_metrics["event_balanced_at_macro_f1_selected"][
                    "macro_f1"
                ],
                places=12,
            )
            self.assertFalse(result["test_or_sealed_read"])


if __name__ == "__main__":
    unittest.main()
