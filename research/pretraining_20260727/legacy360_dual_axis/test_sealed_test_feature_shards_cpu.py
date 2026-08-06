from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sealed_test_feature_shards as utility  # noqa: E402


PATH_COLUMNS = (
    "s2_0_path",
    "s2_90_path",
    "s2_360_path",
    "l89_0_path",
    "l89_90_path",
    "l89_360_path",
    "emit_0_path",
    "emit_90_path",
    "emit_360_path",
    "s5p_0_path",
)


def _row(
    index: int,
    *,
    event: str,
    plume: str,
    signature: str = "s2",
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": f"id-{index}",
        "plume_id": plume,
        "event_id": event,
        "label": index % 2,
        "query360_index": index,
        "availability_signature": signature,
        **{column: "" for column in PATH_COLUMNS},
    }
    for sensor in signature.split("+"):
        if sensor == "s5p":
            row["s5p_0_path"] = f"/synthetic/{index}/s5p_0.npz"
        else:
            for role in ("0", "90", "360"):
                row[f"{sensor}_{role}_path"] = (
                    f"/synthetic/{index}/{sensor}_{role}.tif"
                )
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_selection(
    root: Path,
    *,
    family: str = "gated",
) -> tuple[Path, Path, dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (root / "checkpoint_best.pth").absolute()
    encoder = {
        "default_weights": "/synthetic/panopticon.pth",
        "sensor_weights": {},
        "provenance": {"fixture": True},
        "state_sha256": "a" * 64,
    }
    model_config = {
        "feature_dim": 3,
        "num_sensors": 4,
        "num_roles": 3,
        "bottleneck_dim": 2,
    }
    if family == "gated":
        lock_schema = "query360-gated-delta-selection-lock-v1"
        checkpoint_schema = "query360-gated-delta-head-v1"
        arm = "gated_delta"
    elif family == "two_axis":
        lock_schema = "query360-two-axis-selection-lock-v1"
        checkpoint_schema = "query360-two-axis-head-v1"
        arm = "both"
    else:
        raise AssertionError(family)
    checkpoint = {
        "schema_version": checkpoint_schema,
        "epoch": 2,
        "arm": arm,
        "base_mode": "universal",
        "model_config": model_config,
        "model": {"weight": torch.ones(1)},
        "parameter_signature": {
            "parameter_count": 1,
            "names": ["weight"],
        },
        "dev": {"best_binary_f1": 0.8125},
        "locked_threshold_candidate": 0.4,
    }
    if family == "gated":
        checkpoint["encoder"] = encoder
        checkpoint["base_contract"] = {"mode": "universal"}
    torch.save(checkpoint, checkpoint_path)
    lock = {
        "schema_version": lock_schema,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": utility.sha256_file(checkpoint_path),
        "best_epoch": 2,
        "selection_metric": "best_binary_f1",
        "selection_score": 0.8125,
        "locked_threshold": 0.4,
        "arm": arm,
        "base_mode": "universal",
        "model_config": model_config,
        "encoder": encoder,
        "test_cache_read_before_lock": False,
    }
    if family == "gated":
        lock["base_contract"] = {"mode": "universal"}
    lock_path = (root / "selection_lock.json").absolute()
    utility.atomic_json(lock_path, lock)
    utility.atomic_json(
        root / "summary.json",
        {
            "status": "complete",
            "selection_lock": lock,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    utility.atomic_json(
        root / "run_status.json",
        {
            "status": "complete",
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    return lock_path, checkpoint_path, lock, checkpoint


def _write_master_selection(
    root: Path,
    *,
    lock_path: Path,
    checkpoint_path: Path,
    lock: dict,
    family: str,
) -> Path:
    winner_family = {
        "gated": "gated_delta",
        "two_axis": "compact_axial",
    }[family]
    winner = {
        "run": f"synthetic-{winner_family}-winner",
        "candidate_index": 0,
        "family": winner_family,
        "arm": lock["arm"],
        "base_mode": lock["base_mode"],
        "learning_rate": 1e-4,
        "selection_metric": lock["selection_metric"],
        "selection_score": lock["selection_score"],
        "ap": 0.9,
        "auc": 0.9,
        "best_epoch": lock["best_epoch"],
        "locked_threshold": lock["locked_threshold"],
        "parameter_count": 1,
        "lock_schema": lock["schema_version"],
        "checkpoint_schema": {
            "gated": "query360-gated-delta-head-v1",
            "two_axis": "query360-two-axis-head-v1",
        }[family],
        "selection_lock": {
            "path": str(lock_path.absolute()),
            "sha256": utility.sha256_file(lock_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path.absolute()),
            "sha256": utility.sha256_file(checkpoint_path),
        },
    }
    receipt = {
        "schema_version": utility.MASTER_SELECTION_SCHEMA,
        "script_version": "synthetic-master-selection-v1",
        "status": "complete",
        "common_contract": {
            "encoder": lock["encoder"],
            "encoder_fingerprint": utility.object_fingerprint(
                lock["encoder"]
            ),
        },
        "winner": winner,
        "locked_dispatch": {
            "evaluator_family": winner_family,
            "evaluator": str(
                SCRIPT_DIR.parent
                / (
                    "query360_gated_delta_runner.py"
                    if family == "gated"
                    else "query360_two_axis_full_legacy.py"
                )
            ),
            "selection_lock": winner["selection_lock"],
            "checkpoint": winner["checkpoint"],
        },
        "sealed_test_read": False,
        "sealed_test_evaluations": 0,
    }
    path = (root / "master_dev_selection_receipt.json").absolute()
    utility.atomic_json(path, receipt)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{utility.sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )
    return path


def _shard_args(
    *,
    manifest: Path,
    lock: Path,
    checkpoint: Path,
    master: Path,
    output: Path,
    sealed_test: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=str(manifest),
        selection_lock=str(lock),
        checkpoint=str(checkpoint),
        master_selection_receipt=str(master),
        output_dir=str(output),
        prefix="sealed",
        plan="",
        sealed_test=sealed_test,
        extractor=str(
            SCRIPT_DIR.parent / "query360_two_axis_full_legacy.py"
        ),
        python=sys.executable,
        raw_cache_dir="/synthetic/local-readonly-cache",
        raw_cache_readonly_fallback=True,
        row_batch_size=4,
        encoder_microbatch=8,
        num_workers=2,
        prefetch_factor=2,
    )


def _cache(
    path: Path,
    manifest_path: Path,
    *,
    encoder: dict,
    split: str = "test",
    sealed_test_read: bool = True,
    sealed_test_authorized: bool = True,
    base_marker: str = "same",
) -> None:
    frame = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False, low_memory=False
    )
    rows = len(frame)
    query = torch.tensor(
        frame["query360_index"].astype(int).tolist(), dtype=torch.float32
    )
    features = query[:, None, None, None].expand(rows, 4, 3, 3).clone()
    valid = torch.ones(rows, 4, 3, dtype=torch.bool)
    sensor_logits = torch.zeros(rows, 4)
    sensor_valid = torch.ones(rows, 4, dtype=torch.bool)
    fused = query / 100.0
    payload = {
        "schema_version": utility.FEATURE_SCHEMA,
        "script_version": "synthetic-extractor-v1",
        "split": split,
        "features": features,
        "features_hybrid": features,
        "features_universal": features + 1,
        "valid_mask": valid,
        "base_sensor_logits": sensor_logits,
        "base_sensor_valid": sensor_valid,
        "base_sensor_logits_hybrid": sensor_logits,
        "base_sensor_valid_hybrid": sensor_valid,
        "base_sensor_logits_universal": sensor_logits + 1,
        "base_sensor_valid_universal": sensor_valid,
        "base_fused_logits": fused,
        "base_hybrid_logits": fused,
        "base_universal_logits": fused + 1,
        "base_definitions": {
            "universal": base_marker,
            "hybrid": base_marker,
        },
        "labels": torch.tensor(
            frame["label"].astype(int).tolist(), dtype=torch.long
        ),
        "ids": frame["id"].astype(str).tolist(),
        "plume_ids": frame["plume_id"].astype(str).tolist(),
        "event_ids": frame["event_id"].astype(str).tolist(),
        "availability_signatures": frame[
            "availability_signature"
        ].astype(str).tolist(),
        "query360_indices": frame[
            "query360_index"
        ].astype(int).tolist(),
        "sensor_names": list(utility.SENSOR_NAMES),
        "manifest": {
            "path": str(manifest_path.absolute()),
            "sha256": utility.sha256_file(manifest_path),
            "rows": rows,
        },
        "encoder": encoder,
        "extraction": {
            "sealed_test_authorized": sealed_test_authorized,
            "observations": rows * 12,
        },
        "sealed_test_read": sealed_test_read,
    }
    torch.save(payload, path)


class SealedFeatureShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="sealed-feature-shards-synthetic-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_parser_locks_audited_worker_ceiling(self) -> None:
        args = utility.build_parser().parse_args(
            [
                "shard-manifest",
                "--manifest",
                "/synthetic/manifest.csv",
                "--selection-lock",
                "/synthetic/selection_lock.json",
                "--checkpoint",
                "/synthetic/checkpoint.pth",
                "--master-selection-receipt",
                "/synthetic/master.json",
                "--output-dir",
                "/synthetic/output",
                "--sealed-test",
            ]
        )
        self.assertEqual(args.num_workers, 4)
        self.assertEqual(args.prefetch_factor, 1)

    def _prepare(
        self, name: str = "case", *, family: str = "gated"
    ) -> dict[str, object]:
        root = self.root / name
        selection = root / "selection"
        output = root / "output"
        root.mkdir(parents=True)
        lock, checkpoint, lock_payload, _checkpoint_payload = (
            _write_selection(selection, family=family)
        )
        master = _write_master_selection(
            root,
            lock_path=lock,
            checkpoint_path=checkpoint,
            lock=lock_payload,
            family=family,
        )
        manifest = root / "synthetic_manifest.csv"
        rows = [
            _row(31, event="event-a", plume="plume-a"),
            _row(
                7,
                event="event-a",
                plume="plume-a",
                signature="s2+l89",
            ),
            _row(42, event="event-b", plume="plume-b", signature="emit"),
            _row(5, event="event-c", plume="plume-c", signature="s5p"),
            _row(19, event="event-d", plume="plume-d", signature="l89"),
            _row(11, event="event-e", plume="plume-e"),
        ]
        _write_manifest(manifest, rows)
        utility.command_shard_manifest(
            _shard_args(
                manifest=manifest,
                lock=lock,
                checkpoint=checkpoint,
                master=master,
                output=output,
            )
        )
        plan_path = output / "sealed_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return {
            "root": root,
            "lock": lock,
            "checkpoint": checkpoint,
            "master": master,
            "lock_payload": lock_payload,
            "manifest": manifest,
            "rows": rows,
            "output": output,
            "plan_path": plan_path,
            "plan": plan,
        }

    def _write_caches(
        self,
        prepared: dict[str, object],
        *,
        split0: str = "test",
        encoder1: dict | None = None,
        base1: str = "same",
    ) -> tuple[Path, Path]:
        plan = prepared["plan"]
        assert isinstance(plan, dict)
        encoder = prepared["lock_payload"]["encoder"]
        paths: list[Path] = []
        for index, shard in enumerate(plan["shards"]):
            cache_path = Path(shard["expected_cache_path"])
            _cache(
                cache_path,
                Path(shard["manifest_path"]),
                encoder=(
                    encoder
                    if index == 0 or encoder1 is None
                    else encoder1
                ),
                split=split0 if index == 0 else "test",
                base_marker="same" if index == 0 else base1,
            )
            paths.append(cache_path)
        return paths[0], paths[1]

    def _merge_args(
        self,
        prepared: dict[str, object],
        cache0: Path,
        cache1: Path,
        *,
        sealed_test: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            input_cache=[str(cache0), str(cache1)],
            plan=str(prepared["plan_path"]),
            selection_lock=str(prepared["lock"]),
            checkpoint=str(prepared["checkpoint"]),
            master_selection_receipt=str(prepared["master"]),
            output_cache=str(Path(prepared["output"]) / "merged_test.pt"),
            output_manifest="",
            audit="",
            receipt="",
            sealed_test=sealed_test,
        )

    def test_explicit_flag_and_bad_checkpoint_fail_before_manifest(self) -> None:
        args = _shard_args(
            manifest=self.root / "must-not-open.csv",
            lock=self.root / "missing-lock.json",
            checkpoint=self.root / "missing.pth",
            master=self.root / "missing-master.json",
            output=self.root / "out",
            sealed_test=False,
        )
        with mock.patch.object(
            utility, "read_manifest", side_effect=AssertionError("opened")
        ) as reader:
            with self.assertRaises(PermissionError):
                utility.command_shard_manifest(args)
            reader.assert_not_called()

        selection = self.root / "bad-selection"
        lock, checkpoint, _lock_payload, _checkpoint_payload = (
            _write_selection(selection)
        )
        master = _write_master_selection(
            self.root,
            lock_path=lock,
            checkpoint_path=checkpoint,
            lock=_lock_payload,
            family="gated",
        )
        with checkpoint.open("ab") as stream:
            stream.write(b"tamper")
        args = _shard_args(
            manifest=self.root / "must-not-open.csv",
            lock=lock,
            checkpoint=checkpoint,
            master=master,
            output=self.root / "bad-out",
        )
        with mock.patch.object(
            utility, "read_manifest", side_effect=AssertionError("opened")
        ) as reader:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                utility.command_shard_manifest(args)
            reader.assert_not_called()

    def test_sharding_is_group_complete_exact_and_one_time(self) -> None:
        prepared = self._prepare()
        plan = prepared["plan"]
        shard_frames = [
            pd.read_csv(
                shard["manifest_path"],
                dtype=str,
                keep_default_na=False,
            )
            for shard in plan["shards"]
        ]
        self.assertTrue(
            set(shard_frames[0]["event_id"]).isdisjoint(
                set(shard_frames[1]["event_id"])
            )
        )
        self.assertTrue(
            set(shard_frames[0]["plume_id"]).isdisjoint(
                set(shard_frames[1]["plume_id"])
            )
        )
        self.assertEqual(
            sorted(
                pd.concat(shard_frames)["query360_index"]
                .astype(int)
                .tolist()
            ),
            sorted(row["query360_index"] for row in prepared["rows"]),
        )
        self.assertFalse(plan["metrics_computed"])
        self.assertFalse(plan["threshold_search_performed"])
        for shard in plan["shards"]:
            self.assertTrue(shard["sealed_test_flag_present"])
            self.assertIn("--sealed-test", shard["extract_argv"])
            self.assertEqual(
                shard["extract_argv"][
                    shard["extract_argv"].index("--split") + 1
                ],
                "test",
            )
        claim = json.loads(
            utility._master_manifest_claim_path(
                prepared["master"]
            ).read_text()
        )
        self.assertEqual(claim["status"], "planned")
        with self.assertRaises(FileExistsError):
            utility.command_shard_manifest(
                _shard_args(
                    manifest=prepared["manifest"],
                    lock=prepared["lock"],
                    checkpoint=prepared["checkpoint"],
                    master=prepared["master"],
                    output=prepared["output"],
                )
            )

    def test_merge_restores_source_order_aliases_and_is_one_time(self) -> None:
        prepared = self._prepare()
        cache0, cache1 = self._write_caches(prepared)
        args = self._merge_args(prepared, cache0, cache1)
        utility.command_merge_cache(args)
        output = Path(args.output_cache)
        merged = torch.load(output, map_location="cpu", weights_only=False)
        expected_queries = [
            int(row["query360_index"]) for row in prepared["rows"]
        ]
        self.assertEqual(merged["query360_indices"], expected_queries)
        self.assertEqual(merged["split"], "test")
        self.assertIs(merged["sealed_test_read"], True)
        self.assertIs(merged["features_hybrid"], merged["features"])
        self.assertIs(
            merged["base_fused_logits"], merged["base_hybrid_logits"]
        )
        self.assertFalse(merged["extraction"]["metrics_computed"])
        self.assertFalse(
            merged["extraction"]["threshold_search_performed"]
        )
        manifest = pd.read_csv(
            output.with_suffix(".pt.manifest.csv"),
            dtype=str,
            keep_default_na=False,
        )
        self.assertNotIn(utility.SOURCE_POSITION_COLUMN, manifest.columns)
        self.assertEqual(
            manifest["query360_index"].astype(int).tolist(),
            expected_queries,
        )
        receipt = json.loads(
            output.with_suffix(".pt.receipt.json").read_text()
        )
        self.assertEqual(receipt["sealed_test_evaluations"], 0)
        self.assertFalse(receipt["metrics_computed"])
        claim = json.loads(
            utility._master_manifest_claim_path(
                prepared["master"]
            ).read_text()
        )
        self.assertEqual(claim["status"], "merged")
        args.output_cache = str(Path(prepared["output"]) / "other.pt")
        with self.assertRaises(PermissionError):
            utility.command_merge_cache(args)

    def test_merge_rejects_non_test_cache(self) -> None:
        prepared = self._prepare("wrong-split")
        cache0, cache1 = self._write_caches(
            prepared, split0="train_core"
        )
        with self.assertRaisesRegex(ValueError, "split is not test"):
            utility.command_merge_cache(
                self._merge_args(prepared, cache0, cache1)
            )
        use = json.loads(
            utility._merge_use_path(prepared["master"]).read_text()
        )
        self.assertEqual(use["status"], "failed_after_cache_use_claim")

    def test_merge_rejects_encoder_and_base_mismatch(self) -> None:
        encoder_case = self._prepare("wrong-encoder")
        wrong_encoder = dict(encoder_case["lock_payload"]["encoder"])
        wrong_encoder["state_sha256"] = "b" * 64
        cache0, cache1 = self._write_caches(
            encoder_case, encoder1=wrong_encoder
        )
        with self.assertRaisesRegex(ValueError, "encoder differs"):
            utility.command_merge_cache(
                self._merge_args(encoder_case, cache0, cache1)
            )

        base_case = self._prepare("wrong-base")
        cache0, cache1 = self._write_caches(
            base_case, base1="different"
        )
        with self.assertRaisesRegex(
            ValueError, "base_definitions_fingerprint differs"
        ):
            utility.command_merge_cache(
                self._merge_args(base_case, cache0, cache1)
            )

    def test_merge_rejects_duplicate_inputs_before_cache_claim(self) -> None:
        prepared = self._prepare("duplicate")
        cache0, _cache1 = self._write_caches(prepared)
        with self.assertRaisesRegex(ValueError, "exactly two distinct"):
            utility.command_merge_cache(
                self._merge_args(prepared, cache0, cache0)
            )
        self.assertFalse(
            utility._merge_use_path(prepared["master"]).exists()
        )

    def test_two_axis_legacy_encoder_attestation_is_explicit(self) -> None:
        prepared = self._prepare("legacy", family="two_axis")
        self.assertEqual(
            prepared["plan"]["selection"]["encoder_binding"],
            "sha_locked_legacy_checkpoint_plus_selection_lock",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
