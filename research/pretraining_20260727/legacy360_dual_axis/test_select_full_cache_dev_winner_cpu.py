#!/usr/bin/env python3
"""Synthetic CPU tests for cross-family immutable dev selection."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import select_full_cache_dev_winner as selector  # noqa: E402


COMMON_ENCODER = {
    "state_sha256": "a" * 64,
    "provenance": {"fixture": True},
}
COMMON_TRAIN_MANIFEST = {
    "path": "/synthetic/development/train.csv",
    "sha256": "b" * 64,
    "rows": selector.EXPECTED_TRAIN_ROWS,
}
COMMON_DEV_MANIFEST = {
    "path": "/synthetic/development/dev.csv",
    "sha256": "c" * 64,
    "rows": selector.EXPECTED_DEV_ROWS,
}
COMMON_SPLIT_GUARD = {
    "train_plume_ids": ["train-plume"],
    "dev_plume_ids": ["dev-plume"],
    "source_train_event_ids": ["train-event", "dev-event"],
}


def _metric(
    score: float,
    *,
    ap: float,
    auc: float,
    threshold: float,
) -> dict[str, float]:
    return {
        "best_binary_f1": score,
        "ap": ap,
        "auc": auc,
        "best_binary_f1_threshold": threshold,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_candidate(
    campaign: Path,
    spec: selector.CandidateSpec,
    *,
    score: float,
    ap: float,
    auc: float,
    parameter_count: int,
    epochs: int = 3,
    epoch0: dict[str, float] | None = None,
) -> None:
    run_dir = campaign / spec.name
    run_dir.mkdir(parents=True)
    epoch0_metrics = epoch0 or _metric(
        0.89, ap=0.95, auc=0.95, threshold=0.4
    )
    best_metrics = _metric(score, ap=ap, auc=auc, threshold=0.42)
    history = [
        {
            "epoch": 0,
            "train": None,
            "dev": epoch0_metrics,
            "interpretation": "exact unchanged checkpoint base",
        }
    ]
    for epoch in range(1, epochs + 1):
        history.append(
            {
                "epoch": epoch,
                "train": {"loss": 1.0 / epoch},
                "dev": (
                    best_metrics
                    if epoch == 1
                    else _metric(
                        max(0.0, score - 0.001 * epoch),
                        ap=max(0.0, ap - 0.001 * epoch),
                        auc=max(0.0, auc - 0.001 * epoch),
                        threshold=0.43,
                    )
                ),
            }
        )
    best = history[1] if score > epoch0_metrics["best_binary_f1"] else history[0]
    best_metrics = best["dev"]

    if spec.family == "gated_delta":
        model_config = {
            "feature_dim": 768,
            "num_sensors": 4,
            "num_roles": 3,
            "bottleneck_dim": 32,
            "dropout": 0.05,
            "residual_cap": 1.5,
        }
        training = {
            "seed": 42,
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "batch_size": 1024,
            "learning_rate": spec.learning_rate,
            "weight_decay": 0.01,
            "sensor_aux_weight": 0.1,
            "residual_l2": 1e-3,
            "selection_metric": selector.SELECTION_METRIC,
        }
        model_summary = {
            "arm": spec.arm,
            "base_mode": "universal",
            "config": model_config,
        }
    else:
        model_config = {
            "embed_dim": 768,
            "num_sensors": 4,
            "num_roles": 3,
            "model_dim": 64,
            "num_heads": 4,
            "temporal_depth": 1,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
        }
        training = {
            "seed": 42,
            "epochs": epochs,
            "batch_size": 1024,
            "learning_rate": spec.learning_rate,
            "sensor_aux_weight": 0.05,
            "axis_aux_weight": 0.05,
            "selection_metric": selector.SELECTION_METRIC,
        }
        model_summary = {
            "arm": spec.arm,
            "base_mode": "universal",
            "model_dim": 64,
            "num_heads": 4,
            "temporal_depth": 1,
        }

    checkpoint_path = (run_dir / "checkpoint_best.pth").absolute()
    checkpoint = {
        "schema_version": spec.checkpoint_schema,
        "epoch": int(best["epoch"]),
        "arm": spec.arm,
        "base_mode": "universal",
        "model_config": model_config,
        "model": {"weight": torch.ones(1)},
        "parameter_signature": {
            "parameter_count": parameter_count,
            "parameter_shapes": {},
            "shape_sha256": "d" * 64,
        },
        "dev": best_metrics,
        "locked_threshold_candidate": best_metrics[
            "best_binary_f1_threshold"
        ],
    }
    if spec.family == "gated_delta":
        checkpoint["encoder"] = COMMON_ENCODER
        checkpoint["base_contract"] = {"mode": "universal"}
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    lock = {
        "schema_version": spec.lock_schema,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "best_epoch": int(best["epoch"]),
        "selection_metric": selector.SELECTION_METRIC,
        "selection_score": best_metrics[selector.SELECTION_METRIC],
        "locked_threshold": best_metrics["best_binary_f1_threshold"],
        "arm": spec.arm,
        "base_mode": "universal",
        "model_config": model_config,
        "encoder": COMMON_ENCODER,
        "split_guard": COMMON_SPLIT_GUARD,
        "train_manifest": COMMON_TRAIN_MANIFEST,
        "dev_manifest": COMMON_DEV_MANIFEST,
        "test_cache_read_before_lock": False,
    }
    if spec.family == "gated_delta":
        lock["base_contract"] = {"mode": "universal"}
        lock["dev_only_runner"] = True

    _write_json(
        run_dir / "launcher_status.json",
        {
            "schema_version": selector.RUN_LAUNCH_STATUS_SCHEMA,
            "status": "complete",
            "run": spec.name,
            "family": spec.family,
            "arm": spec.arm,
            "base_mode": "universal",
            "learning_rate": spec.learning_rate,
            "epochs": epochs,
            "epoch0_base_required": True,
            "epoch0_base_verified": True,
            "dev_only": True,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    _write_json(
        run_dir / "run_status.json",
        {
            "status": "complete",
            "best_epoch": int(best["epoch"]),
            "best_dev_metrics": best_metrics,
            "locked_threshold": best_metrics[
                "best_binary_f1_threshold"
            ],
            "dev_only": True,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    _write_json(run_dir / "metrics_history.json", history)
    _write_json(run_dir / "selection_lock.json", lock)
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": spec.summary_schema,
            "status": "complete",
            "train_rows": selector.EXPECTED_TRAIN_ROWS,
            "dev_rows": selector.EXPECTED_DEV_ROWS,
            "model": model_summary,
            "training": training,
            "best": best,
            "selection_lock": lock,
            "history": history,
            "sealed_test": None,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )


def _campaign(
    root: Path,
    *,
    scores: dict[str, tuple[float, float, float, int]] | None = None,
) -> Path:
    campaign = root / "full_cache_dev_promotion"
    campaign.mkdir(parents=True)
    _write_json(
        campaign / "promotion_launcher_status.json",
        {
            "schema_version": selector.CAMPAIGN_STATUS_SCHEMA,
            "status": "complete",
            "candidate_count": 4,
            "epochs": 3,
            "epoch0_base_required": True,
            "dev_only": True,
            "sealed_test_read": False,
            "sealed_test_evaluations": 0,
        },
    )
    default = {
        selector.CANDIDATES[0].name: (0.903, 0.956, 0.957, 51000),
        selector.CANDIDATES[1].name: (0.901, 0.958, 0.959, 51000),
        selector.CANDIDATES[2].name: (0.902, 0.960, 0.961, 165000),
        selector.CANDIDATES[3].name: (0.900, 0.955, 0.956, 165000),
    }
    if scores:
        default.update(scores)
    for spec in selector.CANDIDATES:
        score, ap, auc, parameters = default[spec.name]
        _write_candidate(
            campaign,
            spec,
            score=score,
            ap=ap,
            auc=auc,
            parameter_count=parameters,
        )
    return campaign


class CrossFamilyWinnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="full-cache-dev-winner-synthetic-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_gated_and_refuses_overwrite(self) -> None:
        campaign = _campaign(self.root)
        selector.main(["--campaign-root", str(campaign)])
        receipt_path = campaign / "master_dev_selection_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(
            receipt["winner"]["run"], selector.CANDIDATES[0].name
        )
        self.assertEqual(receipt["winner"]["family"], "gated_delta")
        self.assertTrue(
            receipt["locked_dispatch"]["evaluator"].endswith(
                "query360_gated_delta_runner.py"
            )
        )
        self.assertFalse(receipt["sealed_test_read"])
        self.assertEqual(receipt["sealed_test_evaluations"], 0)
        sidecar = receipt_path.with_suffix(".json.sha256")
        self.assertEqual(
            sidecar.read_text().split()[0],
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        )
        with self.assertRaises(FileExistsError):
            selector.main(["--campaign-root", str(campaign)])

    def test_selects_compact_axial_and_dispatches_axial_evaluator(self) -> None:
        axial = selector.CANDIDATES[2]
        campaign = _campaign(
            self.root,
            scores={
                axial.name: (0.91, 0.96, 0.97, 165000),
            },
        )
        output = self.root / "winner.json"
        selector.main(
            [
                "--campaign-root",
                str(campaign),
                "--output",
                str(output),
            ]
        )
        receipt = json.loads(output.read_text())
        self.assertEqual(receipt["winner"]["run"], axial.name)
        self.assertEqual(receipt["winner"]["family"], "compact_axial")
        self.assertTrue(
            receipt["locked_dispatch"]["evaluator"].endswith(
                "query360_two_axis_full_legacy.py"
            )
        )
        self.assertTrue(
            Path(receipt["locked_dispatch"]["feature_sharder"]).is_file()
        )

    def test_explicit_tie_break_prefers_smaller_then_name(self) -> None:
        equal = {
            spec.name: (
                0.91,
                0.96,
                0.97,
                100 if spec.family == "gated_delta" else 200,
            )
            for spec in selector.CANDIDATES
        }
        campaign = _campaign(self.root, scores=equal)
        output = self.root / "tie.json"
        selector.main(
            [
                "--campaign-root",
                str(campaign),
                "--output",
                str(output),
            ]
        )
        receipt = json.loads(output.read_text())
        expected = selector.CANDIDATES[0].name
        self.assertEqual(receipt["winner"]["run"], expected)
        self.assertEqual(
            receipt["protocol"]["rank_key"],
            "(-best_binary_f1,-ap,-auc,best_epoch,"
            "parameter_count,candidate_index)",
        )

    def test_rejects_checkpoint_tamper_and_positive_heldout_marker(self) -> None:
        tamper_campaign = _campaign(self.root / "tamper")
        checkpoint = (
            tamper_campaign
            / selector.CANDIDATES[0].name
            / "checkpoint_best.pth"
        )
        with checkpoint.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(
            selector.SelectionAuditError, "checkpoint SHA256"
        ):
            selector.main(
                [
                    "--campaign-root",
                    str(tamper_campaign),
                    "--output",
                    str(self.root / "tamper_receipt.json"),
                ]
            )

        heldout_campaign = _campaign(self.root / "heldout")
        status_path = (
            heldout_campaign
            / selector.CANDIDATES[1].name
            / "run_status.json"
        )
        status = json.loads(status_path.read_text())
        status["sealed_test_read"] = True
        _write_json(status_path, status)
        with self.assertRaisesRegex(
            selector.SelectionAuditError, "must be false"
        ):
            selector.main(
                [
                    "--campaign-root",
                    str(heldout_campaign),
                    "--output",
                    str(self.root / "heldout_receipt.json"),
                ]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
