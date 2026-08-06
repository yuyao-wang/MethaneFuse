from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import axisstats_control as axisstats  # noqa: E402


def synthetic_payload(
    rows: int,
    *,
    split: str,
    prefix: str,
    seed: int,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    labels = (torch.arange(rows) % 2).long()
    signal = labels.float() * 2.0 - 1.0
    dimensions = 16
    features_universal = torch.randn(
        rows, 4, 3, dimensions, generator=generator
    )
    features_hybrid = features_universal.clone()
    features_universal[:, :, 0, 0] += signal[:, None] * 0.4
    features_hybrid[:, :, 0, 0] += signal[:, None] * 0.6

    valid = torch.ones(rows, 4, 3, dtype=torch.bool)
    valid[::3, 1, 2] = False
    valid[::4, 2, 1] = False
    valid[:, 3, 1:] = False
    base_sensor_valid = valid[:, :, 0].clone()
    sensor_noise = 0.35 * torch.randn(rows, 4, generator=generator)
    base_sensor_universal = signal[:, None] + sensor_noise
    base_sensor_hybrid = base_sensor_universal.clone()
    base_sensor_hybrid[:, 0] += signal * 0.25
    base_universal = signal + 0.55 * torch.randn(rows, generator=generator)
    base_hybrid = signal + 0.45 * torch.randn(rows, generator=generator)

    availability: list[str] = []
    for row in range(rows):
        present = [
            sensor
            for index, sensor in enumerate(axisstats.SENSORS)
            if bool(valid[row, index, 0])
        ]
        availability.append("+".join(present))

    return {
        "schema_version": axisstats.CACHE_SCHEMA_VERSION,
        "split": split,
        "sealed_test_read": False,
        "features": features_hybrid,
        "features_hybrid": features_hybrid,
        "features_universal": features_universal,
        "valid_mask": valid,
        "base_universal_logits": base_universal,
        "base_hybrid_logits": base_hybrid,
        "base_fused_logits": base_hybrid,
        "base_sensor_logits": base_sensor_hybrid,
        "base_sensor_logits_hybrid": base_sensor_hybrid,
        "base_sensor_logits_universal": base_sensor_universal,
        "base_sensor_valid": base_sensor_valid,
        "base_sensor_valid_hybrid": base_sensor_valid,
        "base_sensor_valid_universal": base_sensor_valid,
        "labels": labels,
        "ids": [f"{prefix}_id_{index}" for index in range(rows)],
        "plume_ids": [
            f"{prefix}_plume_{index // 2}" for index in range(rows)
        ],
        "event_ids": [
            f"{prefix}_event_{index // 4}" for index in range(rows)
        ],
        "query360_indices": [
            index + (0 if prefix == "train" else 100000)
            for index in range(rows)
        ],
        "availability_signatures": availability,
        "sensor_names": list(axisstats.SENSORS),
        "manifest": {
            "path": f"/synthetic/{prefix}_manifest.csv",
            "rows": rows,
            "sha256": "0" * 64,
        },
    }


class AxisStatsControlCpuTest(unittest.TestCase):
    def test_extracts_requested_time_and_sensor_statistics(self) -> None:
        payload = synthetic_payload(
            24, split="train_core", prefix="train", seed=1
        )
        with tempfile.TemporaryDirectory(prefix="axisstats_smoke_") as root:
            cache = Path(root) / "train_core.pt"
            torch.save(payload, cache)
            dataset = axisstats.load_axisstats_cache(cache, "train_core")
        self.assertEqual(dataset.matrix.shape[0], 24)
        self.assertGreater(dataset.matrix.shape[1], 80)
        self.assertTrue(np.isfinite(dataset.matrix).all())
        self.assertIn("base_universal_fused_logit", dataset.feature_names)
        self.assertIn(
            "axis_hybrid_s2_t0_vs_t90_cosine", dataset.feature_names
        )
        self.assertIn(
            "axis_universal_s2_t0_vs_t360_l2_rms",
            dataset.feature_names,
        )
        self.assertIn(
            "axis_hybrid_current_s2_l89_cosine", dataset.feature_names
        )

    def test_rejects_wrong_split_and_forbidden_input_path(self) -> None:
        payload = synthetic_payload(
            24, split="train_core", prefix="train", seed=2
        )
        with tempfile.TemporaryDirectory(prefix="axisstats_smoke_") as root:
            root_path = Path(root)
            cache = root_path / "train_core.pt"
            torch.save(payload, cache)
            with self.assertRaisesRegex(ValueError, "expected=dev"):
                axisstats.load_axisstats_cache(cache, "dev")

            forbidden = root_path / "sealed_cache.pt"
            torch.save(payload, forbidden)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                axisstats.load_axisstats_cache(forbidden, "train_core")

    def test_end_to_end_smoke_writes_lock_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axisstats_smoke_") as root:
            root_path = Path(root)
            train_cache = root_path / "train_core.pt"
            dev_cache = root_path / "canonical_dev.pt"
            output = root_path / "results"
            torch.save(
                synthetic_payload(
                    160, split="train_core", prefix="train", seed=3
                ),
                train_cache,
            )
            dev_payload = synthetic_payload(
                96, split="dev", prefix="dev", seed=4
            )
            query_indices = dev_payload.pop("query360_indices")
            dev_manifest = root_path / "canonical_dev_manifest.csv"
            pd.DataFrame(
                {
                    "id": dev_payload["ids"],
                    "query360_index": query_indices,
                }
            ).to_csv(dev_manifest, index=False)
            dev_payload["manifest"]["path"] = str(dev_manifest)
            torch.save(dev_payload, dev_cache)
            summary = axisstats.run_experiment(
                Namespace(
                    train_cache=str(train_cache),
                    dev_cache=str(dev_cache),
                    output_dir=str(output),
                    grid="smoke",
                    max_iter=20,
                    threads=1,
                    seed=42,
                )
            )
            self.assertFalse(summary["sealed_test_read"])
            self.assertEqual(summary["fitted_candidate_count"], 2)
            for name in (
                "axisstats_best_model.joblib",
                "axisstats_best_full_model.joblib",
                "axisstats_feature_schema.json",
                "best_full_axisstats_by_sensor.json",
                "candidate_metrics.csv",
                "candidate_metrics.json",
                "dev_predictions_all_candidates.csv",
                "dev_predictions_best.csv",
                "selection_lock.json",
                "summary.json",
                "RESULTS.md",
            ):
                self.assertTrue((output / name).is_file(), name)

            lock = json.loads(
                (output / "selection_lock.json").read_text(encoding="utf-8")
            )
            self.assertFalse(lock["protocol"]["sealed_test_read"])
            self.assertFalse(lock["protocol"]["test_or_sealed_input_accepted"])
            self.assertEqual(lock["split_guard"]["event_ids_overlap"], 0)

            model_payload = joblib.load(output / "axisstats_best_model.joblib")
            self.assertEqual(
                model_payload["schema_version"], axisstats.SCHEMA_VERSION
            )
            self.assertIn("predict_proba", dir(model_payload["pipeline"]))


if __name__ == "__main__":
    unittest.main()
