#!/usr/bin/env python3
"""CPU tests for the strict S2-only feature-cache derivation."""

from pathlib import Path
import unittest

import torch

from make_s2_only_feature_cache import make_s2_only_payload


def _payload(*, split: str = "dev", sealed: bool = False):
    rows, sensors, roles, dim = 4, 4, 3, 5
    features = torch.arange(
        rows * sensors * roles * dim, dtype=torch.float16
    ).reshape(rows, sensors, roles, dim)
    valid = torch.ones(rows, sensors, roles, dtype=torch.bool)
    valid[1, 0, 2] = False
    base_valid = torch.ones(rows, sensors, dtype=torch.bool)
    base_valid[2, 0] = False
    base_sensor = torch.arange(rows * sensors, dtype=torch.float32).reshape(
        rows, sensors
    )
    base_hybrid = base_sensor[:, 0].clone()
    return {
        "schema_version": "query360-two-axis-feature-cache-v1",
        "split": split,
        "features": features.clone(),
        "features_hybrid": features.clone(),
        "features_universal": features.clone() + 1,
        "valid_mask": valid,
        "base_sensor_logits": base_sensor.clone(),
        "base_sensor_valid": base_valid.clone(),
        "base_sensor_logits_hybrid": base_sensor.clone(),
        "base_sensor_valid_hybrid": base_valid.clone(),
        "base_sensor_logits_universal": base_sensor.clone() + 1,
        "base_sensor_valid_universal": base_valid.clone(),
        "base_fused_logits": base_hybrid.clone(),
        "base_hybrid_logits": base_hybrid.clone(),
        "base_universal_logits": base_hybrid.clone() + 1,
        "labels": torch.tensor([0, 1, 1, 0]),
        "ids": ["a", "b", "c", "d"],
        "plume_ids": ["pa", "pb", "pc", "pd"],
        "event_ids": ["ea", "eb", "ec", "ed"],
        "availability_signatures": [
            "s2",
            "s2+l89",
            "s2+emit",
            "s2+l89+emit+s5p",
        ],
        "sensor_names": ["s2", "l89", "emit", "s5p"],
        "manifest": {"path": "dev.csv", "sha256": "abc", "rows": rows},
        "encoder": {"state_sha256": "encoder"},
        "sealed_test_read": sealed,
    }


class S2OnlyCacheTest(unittest.TestCase):
    def test_selects_only_exact_three_visit_s2_rows_and_drops_other_axes(self):
        output, audit = make_s2_only_payload(
            _payload(), source_path=Path("/tmp/dev.pt")
        )
        self.assertEqual(output["features"].shape, (2, 1, 3, 5))
        self.assertTrue(bool(output["valid_mask"].all()))
        self.assertEqual(output["sensor_names"], ["s2"])
        self.assertEqual(output["ids"], ["a", "d"])
        self.assertEqual(output["labels"].tolist(), [0, 0])
        self.assertEqual(output["availability_signatures"], ["s2", "s2"])
        self.assertTrue(
            torch.equal(
                output["base_fused_logits"],
                output["base_sensor_logits"][:, 0],
            )
        )
        self.assertNotIn("features_universal", output)
        self.assertNotIn("base_universal_logits", output)
        self.assertEqual(audit["selected_rows"], 2)
        self.assertEqual(audit["dropped_rows"], 2)

    def test_rejects_non_development_splits(self):
        for split, message in (
            ("test", "sealed-test"),
            ("anything_else", "only train_core/dev"),
        ):
            with self.subTest(split=split), self.assertRaisesRegex(
                ValueError, message
            ):
                make_s2_only_payload(
                    _payload(split=split),
                    source_path=Path("/tmp/forbidden.pt"),
                )

    def test_rejects_any_cache_marked_as_sealed_test_read(self):
        with self.assertRaisesRegex(ValueError, "sealed-test"):
            make_s2_only_payload(
                _payload(sealed=True),
                source_path=Path("/tmp/forbidden.pt"),
            )

    def test_allows_explicit_post_lock_test_payload_without_label_audit(self):
        output, audit = make_s2_only_payload(
            _payload(split="test", sealed=True),
            source_path=Path("/tmp/test.pt"),
            expected_split="test",
            allow_sealed_test=True,
        )
        self.assertTrue(output["sealed_test_read"])
        self.assertEqual(audit["labels"], "withheld until locked evaluation")

    def test_fails_if_hybrid_fused_logit_is_not_exact_s2_checkpoint_logit(
        self,
    ):
        payload = _payload()
        payload["base_hybrid_logits"][0] += 0.01
        with self.assertRaisesRegex(RuntimeError, "not the exact S2 head"):
            make_s2_only_payload(
                payload, source_path=Path("/tmp/dev.pt")
            )


if __name__ == "__main__":
    unittest.main()
