#!/usr/bin/env python3
"""CPU-only authorization and contract tests for the L89 outer producer."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from research.tempo_20260728 import tempo_l89_outer_cache_producer as target
from research.tempo_20260728 import (
    tempo_l89_outer_locked_evaluator as evaluator,
)


def produce_args(**updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "confirm": target.PRODUCE_TOKEN,
        "lock_manifest": "/safe/lock.json",
        "authorized_lock_sha256": "lock-sha",
        "producer_spec": "/safe/producer_spec.json",
        "outer_csv": "/official/outer_source.csv",
        "output_dir": "/cache/outer_result",
        "local_image_cache_dir": "/cache/images",
        "device": "cpu",
    }
    values.update(updates)
    return argparse.Namespace(**values)


class L89OuterCacheProducerContracts(unittest.TestCase):
    def test_bad_token_rejects_before_lock_or_outer_path_access(self) -> None:
        args = produce_args(confirm="wrong")
        with (
            mock.patch.object(evaluator, "validate_lock") as validate,
            mock.patch.object(evaluator, "require_outer_path") as outer_path,
        ):
            with self.assertRaisesRegex(RuntimeError, "authorization token"):
                target.produce_once(args)
        validate.assert_not_called()
        outer_path.assert_not_called()

    def test_mismatched_lock_sha_rejects_before_outer_path_access(self) -> None:
        args = produce_args(authorized_lock_sha256="wrong")
        with (
            mock.patch.object(
                evaluator, "validate_lock", return_value=({}, "actual")
            ),
            mock.patch.object(evaluator, "require_outer_path") as outer_path,
        ):
            with self.assertRaisesRegex(RuntimeError, "lock SHA"):
                target.produce_once(args)
        outer_path.assert_not_called()

    def test_target_literal_mismatch_rejects_before_outer_path_access(
        self,
    ) -> None:
        spec_path = "/tmp/producer_spec.json"
        spec_sha = "spec-sha"
        manifest = {
            "protocol": {
                "outer_input": {
                    "producer": {
                        "spec_artifact": {
                            "path": spec_path,
                            "sha256": spec_sha,
                        }
                    }
                }
            }
        }
        spec = {
            "protocol": {"target_csv_literal": "/official/outer_source.csv"}
        }
        args = produce_args(
            producer_spec=spec_path,
            outer_csv="/official/different_outer_source.csv",
        )
        with (
            mock.patch.object(
                evaluator,
                "validate_lock",
                return_value=(manifest, "lock-sha"),
            ),
            mock.patch.object(
                target, "validate_spec", return_value=(spec, spec_sha)
            ),
            mock.patch.object(evaluator, "require_outer_path") as outer_path,
        ):
            with self.assertRaisesRegex(RuntimeError, "literal differs"):
                target.produce_once(args)
        outer_path.assert_not_called()

    def test_build_spec_does_not_require_or_resolve_target_literal(self) -> None:
        forbidden_target = "/path/that/is/not/mounted/official_test.csv"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.bin"
            dev_cache = root / "dev_cache.pt"
            sidecar = root / "sidecar.pt"
            output = root / "producer_spec.json"
            weights.write_bytes(b"frozen-weights")
            dev_cache.write_bytes(b"safe-development-cache")
            sidecar.write_bytes(b"safe-sidecar")
            weights_sha = evaluator.sha256_file(weights)
            contract = {
                "path_columns": [
                    "path_t0",
                    "path_prev1",
                    "path_prev2",
                    "path_prev3",
                    "path_seasonal",
                    "path_year",
                ],
                "time_columns": [
                    "t0_image_time",
                    "prev1_image_time",
                    "prev2_image_time",
                    "prev3_image_time",
                    "seasonal_image_time",
                    "year_image_time",
                ],
                "role_names": [
                    "t0",
                    "prev1",
                    "prev2",
                    "prev3",
                    "seasonal",
                    "year",
                ],
                "band_indices": list(range(7)),
                "channel_ids": [442, 482, 560, 654, 864, 1608, 2203],
                "normalization_mean": [1.0] * 7,
                "normalization_std": [2.0] * 7,
                "normalization_source": "synthetic_safe_development",
                "image_size": 224,
                "min_valid_fraction": 0.75,
                "validity_band_index": 0,
                "zero_invalid_pixels": True,
                "duplicate_rule": "exact_utc",
                "row_selection_seed": 20260727,
            }
            reference = {
                "split": "val",
                "features": torch.zeros(2, 6, 768),
                "input_contract": contract,
                "weights_sha256": weights_sha,
            }
            sidecar_payload = {
                "arm": "p5_correct_response",
                "base_weights_sha256": weights_sha,
                "sidecar_config": {
                    "feature_dim": 768,
                    "response_dim": 16,
                    "rank": 8,
                    "residual_scale": 0.1,
                },
                "objective_response": [0, 0, 0, 0, 0, 0.15, 1.0],
            }
            args = argparse.Namespace(
                output=str(output),
                dev_cache=str(dev_cache),
                weights=str(weights),
                p5_sidecar_checkpoint=str(sidecar),
                target_csv_literal=forbidden_target,
                batch_size=16,
                num_workers=8,
                prefetch_factor=2,
                local_cache_workers=8,
                local_cache_min_free_gb=20.0,
            )
            with (
                mock.patch.object(
                    target.base_runner, "validate_cache_payload"
                ),
                mock.patch.object(
                    target.torch,
                    "load",
                    side_effect=[reference, sidecar_payload],
                ),
                mock.patch.object(
                    evaluator,
                    "require_outer_path",
                    side_effect=AssertionError(
                        "build-spec must not resolve an outer path"
                    ),
                ) as outer_path,
            ):
                target.build_spec(args)
            outer_path.assert_not_called()
            written = evaluator.read_json(output)
            self.assertEqual(
                written["protocol"]["target_csv_literal"], forbidden_target
            )
            self.assertFalse(
                written["protocol"]["target_csv_access_during_spec_build"]
            )

    def test_cache_namespace_carries_the_frozen_six_role_contract(self) -> None:
        extraction = {
            "cache_split_value": "outer",
            "path_columns": [f"path_{index}" for index in range(6)],
            "time_columns": [f"time_{index}" for index in range(6)],
            "label_column": "label",
            "id_column": "id",
            "plume_id_column": "plume_id",
            "event_column": "event_group_id",
            "band_indices": list(range(7)),
            "image_size": 224,
            "min_valid_fraction": 0.75,
            "validity_band_index": 0,
            "zero_invalid_pixels": True,
            "batch_size": 16,
            "num_workers": 8,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "amp_dtype": "bfloat16",
            "storage_dtype": "float16",
            "local_cache_bypass_root": "/diniuvol/yuyao",
            "local_cache_mode": "sync",
            "local_cache_workers": 8,
            "local_cache_min_free_gb": 20.0,
            "max_rows": 0,
            "row_selection_seed": 20260727,
            "max_invalid_t0": 0,
            "max_read_errors": 0,
        }
        spec = {
            "protocol": {
                "extraction": extraction,
                "locked_artifacts": [
                    {
                        "role": "frozen Panopticon weights",
                        "path": "/safe/weights.pth",
                    }
                ],
            }
        }
        namespace = target._cache_namespace(
            spec=spec,
            staged_csv=Path("/safe/source_manifest.csv"),
            base_output=Path("/safe/base_cache.pt"),
            local_image_cache_dir=Path("/safe/images"),
            device="cpu",
        )
        self.assertEqual(namespace.split, "outer")
        self.assertEqual(namespace.path_columns, ",".join(extraction["path_columns"]))
        self.assertEqual(namespace.time_columns, ",".join(extraction["time_columns"]))
        self.assertEqual(namespace.band_indices, "0,1,2,3,4,5,6")
        self.assertEqual(namespace.weights, "/safe/weights.pth")
        self.assertEqual(namespace.max_invalid_t0, 0)
        self.assertEqual(namespace.max_read_errors, 0)
        self.assertFalse(namespace.overwrite)

    def test_template_lock_cannot_authorize_producer(self) -> None:
        protocol = {"placeholder": True}
        manifest = {
            "status": "development_template_not_authorizable",
            "protocol": protocol,
            "protocol_sha256": evaluator.canonical_digest(protocol),
            "test_or_sealed_or_holdout_read": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "LOCK_MANIFEST.json"
            evaluator.atomic_json(path, manifest)
            args = produce_args(lock_manifest=str(path))
            with mock.patch.object(
                evaluator,
                "require_outer_path",
                side_effect=AssertionError("outer path must stay untouched"),
            ) as outer_path:
                with self.assertRaisesRegex(
                    RuntimeError, "template cannot read outer"
                ):
                    target.produce_once(args)
            outer_path.assert_not_called()

    def test_single_pass_staging_receipt_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "staged.bin"
            source.write_bytes(b"one-pass-source")
            receipt = evaluator.stage_once(
                source, destination, role="synthetic_source"
            )
            self.assertEqual(destination.read_bytes(), b"one-pass-source")
            self.assertEqual(receipt["source_content_passes"], 1)
            self.assertEqual(receipt["bytes_read"], len(b"one-pass-source"))
            self.assertEqual(
                receipt["sha256_during_single_pass"],
                evaluator.sha256_file(source),
            )
            with self.assertRaises(FileExistsError):
                evaluator.stage_once(
                    source, destination, role="synthetic_source"
                )


if __name__ == "__main__":
    unittest.main()
