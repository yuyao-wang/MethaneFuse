#!/usr/bin/env python3
"""Dataset-free CPU tests for the L89 ragged four-arm pilot gate."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    evaluate_l89_ragged_gate as gate,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as runner,
)


class L89RaggedGateTests(unittest.TestCase):
    @staticmethod
    def _cache_payload(
        split: str,
        *,
        labels: list[int],
        event_ids: list[str],
    ) -> dict:
        rows = len(labels)
        timepoints = 3
        feature_dim = 8
        generator = torch.Generator().manual_seed(101 if split == "train" else 103)
        features = torch.randn(
            rows, timepoints, feature_dim, generator=generator
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)
        timestamps = torch.tensor(
            [[1_000 + row, 999 + row, 910 + row] for row in range(rows)],
            dtype=torch.int64,
        )
        delta_days = torch.tensor([[0.0, -1.0, -90.0]]).repeat(rows, 1)
        valid = torch.ones(rows, timepoints, dtype=torch.bool)
        duplicate = torch.zeros_like(valid)
        contract = {
            "script_version": runner.SCRIPT_VERSION,
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
            "format_version": runner.CACHE_FORMAT_VERSION,
            "script_version": runner.SCRIPT_VERSION,
            "split": split,
            "features": features,
            "labels": label_tensor,
            "ids": [f"{split}-id-{row:03d}" for row in range(rows)],
            "plume_ids": [f"{event_id}-A" for event_id in event_ids],
            "event_ids": list(event_ids),
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
            "input_contract_sha256": runner.sha256_bytes(
                runner.canonical_json_bytes(contract)
            ),
            "csv_sha256": f"{split}-csv",
            "weights_sha256": "a" * 64,
            "feature_sha256": runner.tensor_sha256(features),
        }

    @classmethod
    def _build_fixture(cls, root: Path) -> Path:
        run_dir = root / "ragged-four-arm"
        run_dir.mkdir(parents=True)
        train_labels = [0, 0, 1, 1] * 4
        train_events = [f"train-event-{index // 2}" for index in range(16)]
        val_labels = [0, 0, 1, 1] * 4
        val_events = [f"val-event-{index // 2}" for index in range(16)]
        train_cache = root / "train-cache.pt"
        val_cache = root / "validation-cache.pt"
        runner.atomic_torch_save(
            train_cache,
            cls._cache_payload(
                "train", labels=train_labels, event_ids=train_events
            ),
        )
        runner.atomic_torch_save(
            val_cache,
            cls._cache_payload("val", labels=val_labels, event_ids=val_events),
        )
        _, _, cache_audit = runner.load_cache_pair(train_cache, val_cache)

        seed = gate.DEFAULT_BOOTSTRAP_SEED
        torch.manual_seed(seed)
        model = runner.RaggedCurrentQueryHead(
            feature_dim=8,
            num_roles=3,
            model_dim=16,
            num_heads=4,
            depth=2,
            mlp_ratio=2.0,
            dropout=0.0,
            periods_days=(1.0, 7.0, 90.0),
            t0_index=0,
        )
        initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        initial_sha = runner.state_dict_sha256(initial_state)
        parameter_signature = runner.model_parameter_signature(model)

        probabilities = {
            "t0_masked": np.array(
                [
                    0.55,
                    0.35,
                    0.45,
                    0.65,
                    0.60,
                    0.40,
                    0.42,
                    0.58,
                    0.52,
                    0.48,
                    0.49,
                    0.51,
                    0.62,
                    0.38,
                    0.46,
                    0.54,
                ]
            ),
            "role_only": np.array(
                [
                    0.50,
                    0.45,
                    0.55,
                    0.60,
                    0.58,
                    0.42,
                    0.47,
                    0.53,
                    0.56,
                    0.44,
                    0.48,
                    0.52,
                    0.59,
                    0.41,
                    0.49,
                    0.51,
                ]
            ),
            "delta_time": np.array(
                [
                    0.08,
                    0.12,
                    0.88,
                    0.92,
                    0.10,
                    0.15,
                    0.85,
                    0.90,
                    0.05,
                    0.20,
                    0.80,
                    0.95,
                    0.18,
                    0.22,
                    0.78,
                    0.82,
                ]
            ),
            "history_shuffle_train": np.array(
                [
                    0.60,
                    0.40,
                    0.52,
                    0.48,
                    0.54,
                    0.46,
                    0.58,
                    0.42,
                    0.57,
                    0.43,
                    0.51,
                    0.49,
                    0.53,
                    0.47,
                    0.56,
                    0.44,
                ]
            ),
        }
        validation_ids = [f"val-id-{row:03d}" for row in range(16)]
        plume_ids = [f"{event_id}-A" for event_id in val_events]
        labels = np.asarray(val_labels, dtype=np.int64)
        best_by_arm = {}
        for arm in gate.ARM_NAMES:
            arm_dir = run_dir / arm
            arm_dir.mkdir()
            metrics = gate.recompute_metrics(labels, probabilities[arm])
            best_by_arm[arm] = {
                "arm": arm,
                "epoch": 1,
                "optimizer_steps": 1,
                "train_rows_seen": len(train_labels),
                "validation": metrics,
            }
            checkpoint = {
                "script_version": runner.SCRIPT_VERSION,
                "arm": arm,
                "epoch": 1,
                "model": copy.deepcopy(initial_state),
                "model_parameter_signature": copy.deepcopy(parameter_signature),
                "initial_state_sha256": initial_sha,
                "cache_audit": copy.deepcopy(cache_audit),
                "validation": copy.deepcopy(metrics),
                "args": {"seed": seed},
            }
            runner.atomic_torch_save(
                arm_dir / gate.CHECKPOINT_NAME, checkpoint
            )
            pd.DataFrame(
                {
                    "id": validation_ids,
                    "plume_id": plume_ids,
                    "event_id": val_events,
                    "label": labels,
                    "probability": probabilities[arm],
                    "prediction_at_0_5": (
                        probabilities[arm] >= gate.FIXED_THRESHOLD
                    ).astype(np.int64),
                    "arm": arm,
                    "epoch": 1,
                }
            ).to_csv(arm_dir / gate.PREDICTION_NAME, index=False)

        run_config = {
            "script_version": runner.SCRIPT_VERSION,
            "arms": list(gate.ARM_NAMES),
            "seed": seed,
            "epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 4,
            "model_dim": 16,
            "num_heads": 4,
            "depth": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "delta_periods_days": [1.0, 7.0, 90.0],
            "train_rows_total": len(train_labels),
            "train_rows_usable": len(train_labels),
            "validation_rows_total": len(val_labels),
            "validation_rows_usable": len(val_labels),
            "parameter_signature": copy.deepcopy(parameter_signature),
            "initial_state_sha256": initial_sha,
            "matched_compute_contract": {
                "same_initial_state": True,
                "same_model_parameter_shapes": True,
                "same_epoch_batches": True,
                "same_optimizer_and_steps": True,
                "delta_encoder_executed_for_every_arm": True,
                "history_shuffle_scope": "training-only-cross-canonical-event",
            },
            "cache_audit": copy.deepcopy(cache_audit),
        }
        runner.atomic_json_write(run_dir / "run_config.json", run_config)
        runner.atomic_json_write(
            run_dir / "summary.json",
            {
                "script_version": runner.SCRIPT_VERSION,
                "best_by_arm": best_by_arm,
                "selection_metric": "validation_ap",
                "sealed_test_read": False,
                "initial_state_sha256": initial_sha,
                "parameter_signature": copy.deepcopy(parameter_signature),
                "cache_audit": copy.deepcopy(cache_audit),
            },
        )
        runner.atomic_json_write(
            run_dir / "run_status.json",
            {
                "status": "complete",
                "arms": list(gate.ARM_NAMES),
                "sealed_test_read": False,
            },
        )
        return run_dir

    @classmethod
    def _convert_fixture_to_emit(cls, run_dir: Path) -> None:
        """Restamp a synthetic L89 fixture as an exact EMIT wrapper run."""

        config_path = run_dir / "run_config.json"
        summary_path = run_dir / "summary.json"
        status_path = run_dir / "run_status.json"
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        train_cache_path = Path(run_config["cache_audit"]["train_cache"])
        validation_cache_path = Path(
            run_config["cache_audit"]["validation_cache"]
        )
        for cache_path in (train_cache_path, validation_cache_path):
            payload = runner.torch_load_trusted(cache_path)
            payload["script_version"] = gate.EMIT_SCRIPT_VERSION
            payload["sensor"] = gate.EMIT_SENSOR
            contract = dict(payload["input_contract"])
            contract["script_version"] = gate.EMIT_SCRIPT_VERSION
            contract["sensor"] = gate.EMIT_SENSOR
            payload["input_contract"] = contract
            payload["input_contract_sha256"] = runner.sha256_bytes(
                runner.canonical_json_bytes(contract)
            )
            runner.atomic_torch_save(cache_path, payload)

        _, _, cache_audit = runner.load_cache_pair(
            train_cache_path, validation_cache_path
        )
        cache_audit.update(
            {
                "cache_script_version": gate.EMIT_SCRIPT_VERSION,
                "sensor": gate.EMIT_SENSOR,
            }
        )
        run_config["script_version"] = gate.EMIT_SCRIPT_VERSION
        run_config["cache_audit"] = copy.deepcopy(cache_audit)
        runner.atomic_json_write(config_path, run_config)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["script_version"] = gate.EMIT_SCRIPT_VERSION
        summary["cache_audit"] = copy.deepcopy(cache_audit)
        runner.atomic_json_write(summary_path, summary)

        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["script_version"] = gate.EMIT_SCRIPT_VERSION
        runner.atomic_json_write(status_path, status)

        for arm in gate.ARM_NAMES:
            checkpoint_path = run_dir / arm / gate.CHECKPOINT_NAME
            checkpoint = runner.torch_load_trusted(checkpoint_path)
            checkpoint["script_version"] = gate.EMIT_SCRIPT_VERSION
            checkpoint["cache_audit"] = copy.deepcopy(cache_audit)
            runner.atomic_torch_save(checkpoint_path, checkpoint)

    def test_01_pairing_metrics_gate_and_atomic_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-gate.") as directory:
            root = Path(directory)
            run_dir = self._build_fixture(root)
            output_json = root / "gate.json"
            output_csv = root / "gate.csv"
            artifact = gate.evaluate_run(
                run_dir,
                output_json=output_json,
                output_csv=output_csv,
                bootstrap_replicates=200,
                bootstrap_seed=gate.DEFAULT_BOOTSTRAP_SEED,
            )
            self.assertTrue(artifact["integrity"]["all_checks_pass"])
            self.assertTrue(artifact["gate_decision"]["pass"])
            self.assertEqual(artifact["status"], "pass")
            self.assertEqual(
                artifact["metrics"]["per_arm"]["delta_time"][
                    "balanced_accuracy_at_0_5"
                ],
                1.0,
            )
            self.assertEqual(
                artifact["metrics"]["per_arm"]["delta_time"][
                    "pred_positive_rate_at_0_5"
                ],
                0.5,
            )
            for comparator in gate.COMPARATOR_ARMS:
                comparison = artifact["metrics"]["comparisons"][
                    f"delta_time_minus_{comparator}"
                ]
                self.assertGreater(comparison["ap"]["delta"], 0.0)
                self.assertGreater(comparison["ap"]["ci_95"]["lower"], 0.0)
                self.assertTrue(comparison["ap"]["ci_lower_gt_zero"])
            exploratory = artifact["exploratory_not_preregistered"]
            self.assertTrue(exploratory["not_part_of_gate"])
            self.assertEqual(
                set(exploratory["comparisons"]),
                {
                    "role_only_minus_t0_masked",
                    "role_only_minus_history_shuffle_train",
                },
            )
            self.assertNotIn(
                "role_only_minus_t0_masked",
                artifact["gate_decision"]["criteria"],
            )
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_csv.is_file())
            self.assertEqual(
                artifact["outputs"]["csv_sha256"],
                runner.sha256_file(output_csv),
            )
            output_frame = pd.read_csv(output_csv)
            self.assertEqual(len(output_frame), 5)
            self.assertEqual(
                set(output_frame["analysis_scope"]),
                {"preregistered", "exploratory_not_preregistered"},
            )
            self.assertFalse(list(root.glob(".gate.*.tmp")))

    def test_02_cluster_plan_is_paired_clustered_and_deterministic(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.int64)
        event_ids = np.asarray(["A", "A", "B", "C", "C", "D", "D", "D"])
        first = gate.build_event_cluster_bootstrap_plan(
            labels,
            event_ids,
            replicates=50,
            seed=37,
        )
        second = gate.build_event_cluster_bootstrap_plan(
            labels,
            event_ids,
            replicates=50,
            seed=37,
        )
        self.assertEqual(first["attempts"], second["attempts"])
        for observed, expected in zip(first["indices"], second["indices"]):
            self.assertTrue(np.array_equal(observed, expected))
            sampled_events = event_ids[observed]
            for event_id in np.unique(event_ids):
                original_rows = int(np.sum(event_ids == event_id))
                sampled_rows = int(np.sum(sampled_events == event_id))
                self.assertEqual(sampled_rows % original_rows, 0)

        probability = np.asarray(
            [0.1, 0.2, 0.8, 0.3, 0.4, 0.7, 0.9, 0.85]
        )
        delta = gate.paired_cluster_bootstrap(
            labels,
            event_ids,
            probability,
            probability.copy(),
            replicates=50,
            seed=37,
        )
        self.assertEqual(delta["ap"]["delta"], 0.0)
        self.assertEqual(delta["ap"]["ci_95"], {"lower": 0.0, "upper": 0.0})
        self.assertEqual(delta["auc"]["ci_95"], {"lower": 0.0, "upper": 0.0})

    def test_03_prediction_row_order_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-gate.") as directory:
            root = Path(directory)
            run_dir = self._build_fixture(root)
            path = run_dir / "role_only" / gate.PREDICTION_NAME
            frame = pd.read_csv(path)
            frame.iloc[[0, 1]] = frame.iloc[[1, 0]].to_numpy()
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "row pairing/order"):
                gate.evaluate_run(
                    run_dir,
                    output_json=root / "gate.json",
                    output_csv=root / "gate.csv",
                    bootstrap_replicates=20,
                    bootstrap_seed=7,
                )

    def test_04_checkpoint_and_cache_tampering_are_rejected(self) -> None:
        tamper_cases = (
            "initial_state",
            "parameter_shape",
            "cache_audit",
            "compute_budget",
        )
        for tamper in tamper_cases:
            with self.subTest(tamper=tamper):
                with tempfile.TemporaryDirectory(
                    prefix="l89-ragged-gate."
                ) as directory:
                    root = Path(directory)
                    run_dir = self._build_fixture(root)
                    checkpoint_path = (
                        run_dir / "t0_masked" / gate.CHECKPOINT_NAME
                    )
                    if tamper == "compute_budget":
                        summary_path = run_dir / "summary.json"
                        summary = json.loads(summary_path.read_text())
                        summary["best_by_arm"]["t0_masked"][
                            "optimizer_steps"
                        ] = 2
                        runner.atomic_json_write(summary_path, summary)
                        expected = "matched-compute"
                    else:
                        checkpoint = runner.torch_load_trusted(
                            checkpoint_path
                        )
                        if tamper == "initial_state":
                            checkpoint["initial_state_sha256"] = "b" * 64
                            expected = "initial-state SHA"
                        elif tamper == "parameter_shape":
                            checkpoint["model_parameter_signature"][
                                "shape_sha256"
                            ] = "c" * 64
                            expected = "parameter signature"
                        else:
                            checkpoint["cache_audit"]["event_overlap"] = 1
                            expected = "cache audit"
                        runner.atomic_torch_save(checkpoint_path, checkpoint)
                    with self.assertRaisesRegex(ValueError, expected):
                        gate.evaluate_run(
                            run_dir,
                            output_json=root / "gate.json",
                            output_csv=root / "gate.csv",
                            bootstrap_replicates=20,
                            bootstrap_seed=7,
                        )

    def test_05_legacy_main_handler_is_sanitized_before_restricted_load(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-gate.") as directory:
            root = Path(directory)
            run_dir = self._build_fixture(root)
            checkpoint_path = (
                run_dir / "history_shuffle_train" / gate.CHECKPOINT_NAME
            )
            checkpoint = runner.torch_load_trusted(checkpoint_path)
            imported_handler_checkpoint_path = (
                run_dir / "t0_masked" / gate.CHECKPOINT_NAME
            )
            imported_handler_checkpoint = runner.torch_load_trusted(
                imported_handler_checkpoint_path
            )
            imported_handler_checkpoint["args"]["handler"] = runner.train_heads
            runner.atomic_torch_save(
                imported_handler_checkpoint_path,
                imported_handler_checkpoint,
            )

            def inert_handler() -> None:
                return None

            legacy_handler = types.FunctionType(
                inert_handler.__code__,
                {"__builtins__": __builtins__},
                "train_heads",
            )
            legacy_handler.__qualname__ = "train_heads"
            legacy_handler.__module__ = "__main__"
            main_module = sys.modules["__main__"]
            previous = getattr(main_module, "train_heads", None)
            had_previous = hasattr(main_module, "train_heads")
            try:
                setattr(main_module, "train_heads", legacy_handler)
                checkpoint["args"]["handler"] = legacy_handler
                runner.atomic_torch_save(checkpoint_path, checkpoint)
            finally:
                if had_previous:
                    setattr(main_module, "train_heads", previous)
                else:
                    delattr(main_module, "train_heads")

            artifact = gate.evaluate_run(
                run_dir,
                output_json=root / "gate.json",
                output_csv=root / "gate.csv",
                bootstrap_replicates=20,
                bootstrap_seed=7,
            )
            evidence = artifact["inputs"]["arms"]["history_shuffle_train"]
            self.assertTrue(
                evidence[
                    "legacy_handler_sanitized_before_weights_only_load"
                ]
            )
            imported_evidence = artifact["inputs"]["arms"]["t0_masked"]
            self.assertTrue(
                imported_evidence[
                    "legacy_handler_sanitized_before_weights_only_load"
                ]
            )

    def test_06_output_cannot_overwrite_evidence_even_with_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-gate.") as directory:
            root = Path(directory)
            run_dir = self._build_fixture(root)
            with self.assertRaisesRegex(ValueError, "protected input"):
                gate.evaluate_run(
                    run_dir,
                    output_json=run_dir / "run_config.json",
                    output_csv=root / "gate.csv",
                    bootstrap_replicates=20,
                    bootstrap_seed=7,
                    overwrite=True,
                )

    def test_07_ci_lower_is_reported_but_does_not_change_gate(self) -> None:
        per_arm = {
            arm: {
                "balanced_accuracy_at_0_5": 0.75,
                "pred_positive_rate_at_0_5": 0.5,
            }
            for arm in gate.ARM_NAMES
        }
        deltas = {
            "t0_masked": 0.010,
            "role_only": 0.005,
            "history_shuffle_train": 0.005,
        }
        comparisons = {
            f"delta_time_minus_{arm}": {
                "ap": {
                    "delta": delta,
                    "ci_lower_gt_zero": False,
                },
                "auc": {
                    "ci_lower_gt_zero": False,
                },
            }
            for arm, delta in deltas.items()
        }
        decision = gate._evaluate_gate(per_arm, comparisons)
        self.assertTrue(decision["pass"])
        self.assertFalse(
            decision["ci_diagnostics"]["part_of_preregistered_gate"]
        )

    def test_08_legacy_torch_archive_is_rejected_before_load(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-gate.") as directory:
            root = Path(directory)
            run_dir = self._build_fixture(root)
            checkpoint_path = (
                run_dir / "delta_time" / gate.CHECKPOINT_NAME
            )
            checkpoint = runner.torch_load_trusted(checkpoint_path)
            torch.save(
                checkpoint,
                checkpoint_path,
                _use_new_zipfile_serialization=False,
            )
            with self.assertRaisesRegex(ValueError, "modern ZIP"):
                gate.evaluate_run(
                    run_dir,
                    output_json=root / "gate.json",
                    output_csv=root / "gate.csv",
                    bootstrap_replicates=20,
                    bootstrap_seed=7,
                )

    def test_09_exact_emit_cache_provenance_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="l89-ragged-gate.") as directory:
            root = Path(directory)
            run_dir = self._build_fixture(root)
            self._convert_fixture_to_emit(run_dir)
            artifact = gate.evaluate_run(
                run_dir,
                output_json=root / "gate.json",
                output_csv=root / "gate.csv",
                bootstrap_replicates=20,
                bootstrap_seed=7,
            )
            self.assertEqual(
                artifact["runner_script_version"],
                gate.EMIT_SCRIPT_VERSION,
            )
            self.assertEqual(
                set(artifact["inputs"]["cache_audit"]),
                gate.CACHE_AUDIT_KEYS
                | gate.EMIT_CACHE_AUDIT_PROVENANCE_KEYS,
            )
            self.assertEqual(
                artifact["inputs"]["cache_audit"]["sensor"],
                gate.EMIT_SENSOR,
            )

    def test_10_l89_and_emit_cache_audit_schemas_remain_closed(self) -> None:
        cases = (
            ("l89_emit_keys", False, "sensor", "schema mismatch"),
            ("emit_unknown_key", True, "unexpected", "schema mismatch"),
            ("emit_wrong_sensor", True, "sensor", r"sensor must equal"),
            (
                "emit_wrong_cache_version",
                True,
                "cache_script_version",
                r"cache_script_version must equal",
            ),
        )
        for name, emit, key, expected in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory(
                    prefix="l89-ragged-gate."
                ) as directory:
                    root = Path(directory)
                    run_dir = self._build_fixture(root)
                    if emit:
                        self._convert_fixture_to_emit(run_dir)
                    config_path = run_dir / "run_config.json"
                    config = json.loads(
                        config_path.read_text(encoding="utf-8")
                    )
                    if name == "l89_emit_keys":
                        config["cache_audit"].update(
                            {
                                "cache_script_version": (
                                    gate.EMIT_SCRIPT_VERSION
                                ),
                                "sensor": gate.EMIT_SENSOR,
                            }
                        )
                    elif key == "sensor":
                        config["cache_audit"][key] = "not-emit32"
                    elif key == "cache_script_version":
                        config["cache_audit"][key] = "not-an-emit-version"
                    else:
                        config["cache_audit"][key] = "not-allowed"
                    runner.atomic_json_write(config_path, config)
                    with self.assertRaisesRegex(ValueError, expected):
                        gate.evaluate_run(
                            run_dir,
                            output_json=root / "gate.json",
                            output_csv=root / "gate.csv",
                            bootstrap_replicates=20,
                            bootstrap_seed=7,
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
