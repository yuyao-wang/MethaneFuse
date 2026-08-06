#!/usr/bin/env python3
"""CPU-only static contracts for the immutable clean-L89 launcher."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).with_name("run_l89_clean_replicate.sh")


def assignment(script: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}=([0-9]+)$", script, re.MULTILINE)
    if match is None:
        raise AssertionError(f"Missing integer assignment {name}.")
    return int(match.group(1))


class LauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = LAUNCHER.read_text(encoding="utf-8")

    def test_exact_frozen_split_shas_are_hard_locked(self) -> None:
        for digest in (
            "8cee38e78f0ff7ebf5e34c02ba07d351cbf93438d9a78e88ad741dc8be3a22eb",
            "8caa5f7a31340c5007ec4457fd5766179a23dc1e97b432322768a8cef0144883",
            "da54079d348d3d9cf13d3609a61e6315169e75d8b8ee977caa8143901c7c69df",
            "3e11d6795c7da28a2c379e51dd993c049e6bb9af37e79b45aae1d9140a68bfec",
        ):
            self.assertIn(digest, self.script)
        self.assertEqual(self.script.count("verify_exact_sha256 \\\n"), 4)
        self.assertIn('"$source_inner_readiness"', self.script)

    def test_gpu_threshold_covers_measured_pair_peak_and_margin(self) -> None:
        peak = assignment(self.script, "measured_dual_extractor_peak_mib")
        margin = assignment(self.script, "gpu0_safety_margin_mib")
        threshold = assignment(self.script, "minimum_gpu0_free_mib")
        self.assertEqual(peak, 44_671)
        self.assertGreaterEqual(margin, 8_192)
        self.assertGreaterEqual(threshold, peak + margin)
        self.assertGreaterEqual(threshold, 53_248)

    def test_second_resource_gate_is_adjacent_to_base_pair(self) -> None:
        initial = self.script.index('preflight_resources "initial"')
        second = self.script.index('preflight_resources "base-pair"')
        sampler = self.script.index(
            'start_sampler "$log_root/gpu_base_cache.csv"', second
        )
        pair = self.script.index(
            "run_pair base-train", sampler
        )
        self.assertLess(initial, second)
        self.assertLess(second, sampler)
        self.assertLess(sampler, pair)
        between = self.script[second:sampler]
        self.assertNotIn("run_pair ", between)
        self.assertNotIn('"${cmd_base_', between)

    def test_frozen_source_inputs_are_in_end_of_run_closure(self) -> None:
        for name in (
            '"$source_inner_train"',
            '"$source_inner_dev"',
            '"$source_inner_readiness"',
            '"$staging_complete_receipt"',
        ):
            self.assertIn(name, self.script)
        self.assertIn(
            'sha256sum -c "$provenance_root/INPUT_SHA256SUMS.txt"',
            self.script,
        )

    def test_promotion_recomputes_fresh_p4_predictions(self) -> None:
        promotion_start = self.script.index("cmd_promotion=(")
        promotion_end = self.script.index("\n)", promotion_start)
        command = self.script[promotion_start:promotion_end]
        self.assertIn("--p0-predictions", command)
        self.assertIn(
            '"$d1_root/seed_20260727/p0_base_predictions.csv"', command
        )
        self.assertIn("--p0-head-predictions", command)
        self.assertIn(
            '"$head_root/p0/validation_best_event_balanced_ap_predictions.csv"',
            command,
        )
        self.assertIn("--p4-predictions", command)
        self.assertIn(
            '"$head_root/p4/validation_best_event_balanced_ap_predictions.csv"',
            command,
        )
        self.assertIn("--p5-predictions", command)

    def test_p4_and_d1_receipts_are_in_artifact_closure(self) -> None:
        closure_start = self.script.index("artifact_files=(")
        closure_end = self.script.index("\n)", closure_start)
        closure = self.script[closure_start:closure_end]
        self.assertIn(
            '"$head_root/p4/validation_best_event_balanced_ap_predictions.csv"',
            closure,
        )
        for seed in (20260727, 20260728, 20260729):
            self.assertIn(
                f'"$d1_root/seed_{seed}/p0_base_predictions.csv"',
                closure,
            )
            self.assertIn(
                f'"$d1_root/seed_{seed}/summary.json"',
                closure,
            )
            self.assertIn(
                f'"$d1_root/seed_{seed}/d1_gated_delta_metrics_history.json"',
                closure,
            )


if __name__ == "__main__":
    unittest.main()
