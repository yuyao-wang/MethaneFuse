#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import availability_calibration_oof as module


class AvailabilityCalibrationOOFTest(unittest.TestCase):
    def test_exact_threshold_maximizes_binary_f1(self) -> None:
        labels = np.array([1, 0, 1, 0], dtype=np.int8)
        probabilities = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)
        threshold, score = module.best_f1_threshold(labels, probabilities)
        self.assertAlmostEqual(threshold, 0.7)
        self.assertAlmostEqual(score, 0.8)

    def test_forbidden_input_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sealed_test_dev_predictions.csv"
            path.write_text("id\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev-only guard"):
                module.assert_dev_only_input(path, "predictions")

    def test_strict_join_and_event_grouped_oof(self) -> None:
        rows = []
        for event_index in range(10):
            for row_index in range(4):
                label = (event_index + row_index) % 2
                identifier = event_index * 10 + row_index
                signature = "s2+l89" if row_index % 2 else "s2"
                rows.append(
                    {
                        "id": identifier,
                        "plume_id": f"plume-{event_index}",
                        "event_id": f"event-{event_index}",
                        "availability_signature": signature,
                        "label": label,
                        "anchor_sensor": "s2",
                        "query360_index": 1000 + identifier,
                        "probability": 0.8 if label else 0.2,
                    }
                )
        full = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path = root / "toy_dev_predictions.csv"
            manifest_path = root / "toy_dev_manifest.csv"
            full[
                [
                    "id",
                    "plume_id",
                    "availability_signature",
                    "label",
                    "probability",
                ]
            ].to_csv(prediction_path, index=False)
            full[
                [
                    "id",
                    "plume_id",
                    "event_id",
                    "availability_signature",
                    "label",
                    "anchor_sensor",
                    "query360_index",
                ]
            ].to_csv(manifest_path, index=False)

            frame, audit = module.read_and_join(prediction_path, manifest_path)
            assignments, fold_audit = module.build_folds(frame, folds=5, seed=7)
            config = SimpleNamespace(
                min_group_rows=4,
                min_group_class_rows=1,
                min_group_events=2,
                shrink_tau=4.0,
            )
            oof, _ = module.make_oof_predictions(frame, assignments, config)

        self.assertEqual(audit["join_cardinality"], "one_to_one")
        self.assertEqual(audit["event_count"], 10)
        self.assertTrue(all(item["event_overlap"] == 0 for item in fold_audit))
        self.assertTrue(all(item["plume_overlap"] == 0 for item in fold_audit))
        prediction_columns = [
            column for column in oof.columns if column.startswith("prediction_")
        ]
        self.assertTrue(oof[prediction_columns].isin([0, 1]).all().all())

    def test_metadata_disagreement_fails_closed(self) -> None:
        prediction = pd.DataFrame(
            {
                "id": [1],
                "plume_id": ["wrong"],
                "availability_signature": ["s2"],
                "label": [1],
                "probability": [0.9],
            }
        )
        manifest = pd.DataFrame(
            {
                "id": [1],
                "plume_id": ["canonical"],
                "event_id": ["event"],
                "availability_signature": ["s2"],
                "label": [1],
                "anchor_sensor": ["s2"],
                "query360_index": [11],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path = root / "toy_dev_predictions.csv"
            manifest_path = root / "toy_dev_manifest.csv"
            prediction.to_csv(prediction_path, index=False)
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                module.read_and_join(prediction_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
