#!/usr/bin/env python3
"""Evaluate the sole promoted full-cache winner exactly once.

This is the cross-family gate between the immutable development-only master
selection receipt, the one-time sealed feature merge receipt, and the existing
family-specific ``evaluate-locked`` implementation.  It intentionally has no
training, model-selection, or threshold-selection arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_DIR = SCRIPT_DIR.parent
REPO_ROOT = RUNNER_DIR.parent.parent
for import_root in (SCRIPT_DIR, RUNNER_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import sealed_test_feature_shards as sealed  # noqa: E402


SCRIPT_VERSION = "legacy360-promoted-locked-eval-v1"
EVAL_CLAIM_SCHEMA = "legacy360-promoted-locked-eval-use-v1"
EVAL_RECEIPT_SCHEMA = "legacy360-promoted-locked-eval-receipt-v1"
FAMILY_EVALUATORS = {
    "gated_delta": RUNNER_DIR / "query360_gated_delta_runner.py",
    "compact_axial": RUNNER_DIR / "query360_two_axis_full_legacy.py",
}


def _absolute(value: str | Path) -> Path:
    return Path(value).expanduser().absolute()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} is unavailable: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{role} is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_claim_path(master_receipt_path: Path) -> Path:
    return Path(
        str(master_receipt_path) + ".sealed_locked_eval_use.json"
    )


def _read_master_envelope(
    master_receipt_path: Path,
) -> dict[str, Any]:
    """Authenticate the receipt before following its artifact paths."""

    if master_receipt_path.is_symlink():
        raise ValueError("master selection receipt cannot be a symlink")
    receipt = _json(master_receipt_path, "master selection receipt")
    receipt_sha = _sha256(master_receipt_path)
    sidecar = master_receipt_path.with_suffix(
        master_receipt_path.suffix + ".sha256"
    )
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"master selection receipt SHA sidecar is missing: {sidecar}"
        )
    tokens = sidecar.read_text(encoding="utf-8").split()
    if (
        len(tokens) < 2
        or tokens[0] != receipt_sha
        or tokens[1] != master_receipt_path.name
    ):
        raise ValueError("master selection receipt SHA sidecar mismatch")
    if (
        receipt.get("schema_version") != sealed.MASTER_SELECTION_SCHEMA
        or receipt.get("status") != "complete"
    ):
        raise ValueError("master selection receipt is not complete/supported")
    if (
        receipt.get("sealed_test_read") is not False
        or int(receipt.get("sealed_test_evaluations", -1)) != 0
    ):
        raise PermissionError("master selection receipt is not dev-only")
    winner = receipt.get("winner")
    if not isinstance(winner, Mapping):
        raise ValueError("master selection receipt has no winner")
    lock_binding = winner.get("selection_lock")
    checkpoint_binding = winner.get("checkpoint")
    if not isinstance(lock_binding, Mapping) or not isinstance(
        checkpoint_binding, Mapping
    ):
        raise ValueError("master winner lacks lock/checkpoint binding")
    lock_path = _absolute(str(lock_binding.get("path", "")))
    checkpoint_path = _absolute(str(checkpoint_binding.get("path", "")))
    if lock_path.is_symlink() or checkpoint_path.is_symlink():
        raise ValueError("locked selection artifacts cannot be symlinks")
    return {
        "receipt": receipt,
        "sha256": receipt_sha,
        "sidecar": sidecar,
        "selection_lock_path": lock_path,
        "checkpoint_path": checkpoint_path,
    }


def _preflight_family_model(
    *,
    family: str,
    artifacts: Mapping[str, Any],
) -> None:
    """Strict-load the promoted head before a sealed cache is touched."""

    lock_path = Path(str(artifacts["selection_lock_path"]))
    checkpoint_path = Path(str(artifacts["checkpoint_path"]))
    if family == "gated_delta":
        import query360_gated_delta_runner as gated

        lock, _checkpoint, _model, checkpoint_sha, lock_sha = (
            gated._validate_locked_checkpoint(
                selection_lock_path=lock_path,
                checkpoint_path=checkpoint_path,
            )
        )
        if (
            checkpoint_sha != artifacts["checkpoint_sha256"]
            or lock_sha != artifacts["selection_lock_sha256"]
            or lock.get("encoder") != artifacts["encoder"]
        ):
            raise ValueError("gated preflight differs from promoted artifacts")
        return
    if family == "compact_axial":
        import query360_two_axis_full_legacy as axial

        model = axial._build_locked_model(artifacts["lock"]["model_config"])
        if axial.model_parameter_signature(model) != artifacts[
            "checkpoint"
        ].get("parameter_signature"):
            raise ValueError("compact-axial parameter signature differs")
        model.load_state_dict(
            artifacts["checkpoint"]["model"],
            strict=True,
        )
        return
    raise ValueError(f"unsupported promoted family: {family!r}")


def _validate_master_and_model(
    master_receipt_path: Path,
) -> dict[str, Any]:
    envelope = _read_master_envelope(master_receipt_path)
    artifacts = sealed._validate_selection_artifacts(
        selection_lock_path=envelope["selection_lock_path"],
        checkpoint_path=envelope["checkpoint_path"],
    )
    master = sealed._validate_master_selection_receipt(
        master_receipt_path=master_receipt_path,
        artifacts=artifacts,
    )
    receipt = master["receipt"]
    dispatch = receipt["locked_dispatch"]
    family = master["family"]
    expected_evaluator = FAMILY_EVALUATORS.get(family)
    if expected_evaluator is None or not expected_evaluator.is_file():
        raise ValueError("promoted evaluator family is unavailable")
    if _absolute(str(dispatch.get("evaluator", ""))) != expected_evaluator:
        raise ValueError("master evaluator path differs from audited family")
    _preflight_family_model(family=family, artifacts=artifacts)
    return {
        **master,
        "artifacts": artifacts,
        "evaluator": expected_evaluator,
    }


def _validate_merge_receipt(
    *,
    merge_receipt_path: Path,
    master_receipt_path: Path,
    master: Mapping[str, Any],
    test_cache_path: Path,
) -> dict[str, Any]:
    if merge_receipt_path.is_symlink():
        raise ValueError("sealed merge receipt cannot be a symlink")
    receipt = _json(merge_receipt_path, "sealed merge receipt")
    if (
        receipt.get("schema_version") != sealed.MERGE_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
    ):
        raise ValueError("sealed merge receipt is not complete/supported")
    if (
        receipt.get("sealed_test_read") is not True
        or int(receipt.get("sealed_test_evaluations", -1)) != 0
        or receipt.get("metrics_computed") is not False
        or receipt.get("threshold_search_performed") is not False
    ):
        raise PermissionError("sealed merge receipt violates guardrails")
    artifacts = master["artifacts"]
    if (
        _absolute(
            str(receipt.get("master_selection_receipt_path", ""))
        )
        != master_receipt_path
        or receipt.get("master_selection_receipt_sha256")
        != master["sha256"]
        or receipt.get("selection_lock_sha256")
        != artifacts["selection_lock_sha256"]
        or receipt.get("checkpoint_sha256")
        != artifacts["checkpoint_sha256"]
        or _absolute(str(receipt.get("output_cache", "")))
        != test_cache_path
    ):
        raise ValueError("sealed merge receipt differs from promoted winner")
    merge_use_path = _absolute(
        str(receipt.get("one_time_merge_use_path", ""))
    )
    use = _json(merge_use_path, "one-time sealed merge use")
    if (
        use.get("schema_version") != sealed.MERGE_USE_SCHEMA
        or use.get("status") != "complete"
        or _absolute(str(use.get("receipt_path", "")))
        != merge_receipt_path
        or use.get("receipt_sha256") != _sha256(merge_receipt_path)
        or _absolute(str(use.get("output_cache", ""))) != test_cache_path
        or use.get("output_cache_sha256")
        != receipt.get("output_cache_sha256")
        or use.get("master_selection_receipt_sha256")
        != master["sha256"]
        or use.get("selection_lock_sha256")
        != artifacts["selection_lock_sha256"]
        or use.get("checkpoint_sha256")
        != artifacts["checkpoint_sha256"]
    ):
        raise ValueError("one-time sealed merge use differs from receipt")
    return {
        "receipt": receipt,
        "sha256": _sha256(merge_receipt_path),
        "use": use,
        "use_path": merge_use_path,
    }


def _validate_cache_after_claim(
    *,
    test_cache_path: Path,
    merge: Mapping[str, Any],
    master_receipt_path: Path,
    master: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash and open the sealed cache only after the global eval claim."""

    if test_cache_path.is_symlink() or not test_cache_path.is_file():
        raise FileNotFoundError(
            f"sealed feature cache is unavailable/non-regular: "
            f"{test_cache_path}"
        )
    receipt = merge["receipt"]
    cache_sha = _sha256(test_cache_path)
    if cache_sha != receipt.get("output_cache_sha256"):
        raise ValueError("sealed feature cache SHA256 differs from merge receipt")

    manifest_path = _absolute(str(receipt.get("output_manifest", "")))
    audit_path = _absolute(str(receipt.get("audit", "")))
    if (
        _sha256(manifest_path) != receipt.get("output_manifest_sha256")
        or _sha256(audit_path) != receipt.get("audit_sha256")
    ):
        raise ValueError("sealed merge manifest/audit SHA256 mismatch")
    audit = _json(audit_path, "sealed merge audit")
    if (
        audit.get("schema_version") != sealed.MERGE_AUDIT_SCHEMA
        or audit.get("metrics_computed") is not False
        or audit.get("threshold_search_performed") is not False
    ):
        raise ValueError("sealed merge audit violates guardrails")

    payload = torch.load(
        test_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("sealed feature cache is not a mapping")
    extraction = payload.get("extraction")
    artifacts = master["artifacts"]
    if (
        payload.get("schema_version") != sealed.FEATURE_SCHEMA
        or payload.get("split") != "test"
        or payload.get("sealed_test_read") is not True
        or not isinstance(extraction, Mapping)
        or extraction.get("sealed_test_authorized") is not True
        or extraction.get("metrics_computed") is not False
        or extraction.get("threshold_search_performed") is not False
        or extraction.get("master_selection_receipt_path")
        != str(master_receipt_path)
        or extraction.get("master_selection_receipt_sha256")
        != master["sha256"]
        or extraction.get("selection_lock_sha256")
        != artifacts["selection_lock_sha256"]
        or extraction.get("checkpoint_sha256")
        != artifacts["checkpoint_sha256"]
        or payload.get("encoder") != artifacts["encoder"]
    ):
        raise ValueError("sealed feature cache differs from locked promotion")
    manifest = payload.get("manifest")
    if (
        not isinstance(manifest, Mapping)
        or _absolute(str(manifest.get("path", ""))) != manifest_path
        or manifest.get("sha256") != receipt.get("output_manifest_sha256")
    ):
        raise ValueError("sealed cache manifest differs from merge receipt")
    return {
        "sha256": cache_sha,
        "rows": int(payload["labels"].shape[0]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(manifest["sha256"]),
        "audit_path": str(audit_path),
        "audit_sha256": str(receipt["audit_sha256"]),
    }


def _validate_evaluator_outputs(
    *,
    output_dir: Path,
    test_cache_path: Path,
    master: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = output_dir / "sealed_test_result.json"
    predictions_path = output_dir / "sealed_test_predictions.csv"
    status_path = output_dir / "locked_eval_status.json"
    status = _json(status_path, "locked evaluation status")
    result = _json(result_path, "locked evaluation result")
    artifacts = master["artifacts"]
    checkpoint = result.get("checkpoint")
    selection_lock = result.get("selection_lock")
    cache = result.get("cache")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint, selection_lock, cache)
    ):
        raise ValueError("locked evaluation result is malformed")
    assert isinstance(checkpoint, Mapping)
    assert isinstance(selection_lock, Mapping)
    assert isinstance(cache, Mapping)
    if (
        status.get("status") != "complete"
        or status.get("sealed_test_read") is not True
        or int(status.get("sealed_test_evaluations", -1)) != 1
        or result.get("artifact_type") != "sealed_test_result"
        or int(result.get("evaluation_count", -1)) != 1
        or result.get("test_threshold_search_performed") is not False
        or result.get("test_cache_read_after_selection_lock") is not True
        or checkpoint.get("sha256") != artifacts["checkpoint_sha256"]
        or selection_lock.get("sha256")
        != artifacts["selection_lock_sha256"]
        or float(selection_lock.get("locked_threshold", math.nan))
        != float(artifacts["lock"]["locked_threshold"])
        or _absolute(str(cache.get("path", ""))) != test_cache_path
        or not predictions_path.is_file()
    ):
        raise ValueError("family evaluator output violates locked protocol")
    return {
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "status_path": str(status_path),
        "status_sha256": _sha256(status_path),
        "metrics": result.get("metrics"),
    }


def command_evaluate(args: argparse.Namespace) -> None:
    if not args.sealed_test:
        raise PermissionError(
            "refusing promoted evaluation without explicit --sealed-test"
        )
    if int(args.eval_batch_size) < 1:
        raise ValueError("--eval-batch-size must be positive")

    master_receipt_path = _absolute(args.master_selection_receipt)
    merge_receipt_path = _absolute(args.merge_receipt)
    test_cache_path = _absolute(args.test_cache)
    output_dir = _absolute(args.output_dir)
    master = _validate_master_and_model(master_receipt_path)
    merge = _validate_merge_receipt(
        merge_receipt_path=merge_receipt_path,
        master_receipt_path=master_receipt_path,
        master=master,
        test_cache_path=test_cache_path,
    )

    result_path = output_dir / "sealed_test_result.json"
    predictions_path = output_dir / "sealed_test_predictions.csv"
    status_path = output_dir / "locked_eval_status.json"
    promotion_receipt_path = output_dir / "promotion_eval_receipt.json"
    promotion_sidecar_path = promotion_receipt_path.with_suffix(
        promotion_receipt_path.suffix + ".sha256"
    )
    claim_path = _eval_claim_path(master_receipt_path)
    sealed.refuse_existing(
        (
            result_path,
            predictions_path,
            status_path,
            promotion_receipt_path,
            promotion_sidecar_path,
            claim_path,
        ),
        overwrite=False,
    )

    artifacts = master["artifacts"]
    claim = {
        "schema_version": EVAL_CLAIM_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "status": "claimed_before_test_cache_read",
        "claimed_utc": _utc_now(),
        "master_selection_receipt_path": str(master_receipt_path),
        "master_selection_receipt_sha256": master["sha256"],
        "master_winner_run": master["winner_run"],
        "master_winner_family": master["family"],
        "selection_lock_path": artifacts["selection_lock_path"],
        "selection_lock_sha256": artifacts["selection_lock_sha256"],
        "checkpoint_path": artifacts["checkpoint_path"],
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "locked_threshold": artifacts["lock"]["locked_threshold"],
        "encoder": artifacts["encoder"],
        "encoder_fingerprint": artifacts["encoder_fingerprint"],
        "merge_receipt_path": str(merge_receipt_path),
        "merge_receipt_sha256": merge["sha256"],
        "test_cache_path": str(test_cache_path),
        "expected_test_cache_sha256": merge["receipt"][
            "output_cache_sha256"
        ],
        "output_dir": str(output_dir),
        "sealed_test_read": False,
        "sealed_test_evaluations": 0,
        "threshold_search_performed": False,
    }
    sealed._exclusive_json(claim_path, claim)
    cache_read = False
    evaluator_started = False
    try:
        cache_read = True
        cache = _validate_cache_after_claim(
            test_cache_path=test_cache_path,
            merge=merge,
            master_receipt_path=master_receipt_path,
            master=master,
        )
        dispatch_claim = {
            **claim,
            "status": "dispatching_family_evaluator",
            "sealed_test_read": True,
            "validated_test_cache_sha256": cache["sha256"],
            "validated_test_rows": cache["rows"],
            "evaluator": str(master["evaluator"]),
            "evaluator_sha256": _sha256(master["evaluator"]),
        }
        sealed.atomic_json(claim_path, dispatch_claim)
        argv = [
            str(args.python),
            str(master["evaluator"]),
            "evaluate-locked",
            "--checkpoint",
            artifacts["checkpoint_path"],
            "--selection-lock",
            artifacts["selection_lock_path"],
            "--test-cache",
            str(test_cache_path),
            "--output-dir",
            str(output_dir),
            "--sealed-test",
            "--eval-batch-size",
            str(int(args.eval_batch_size)),
            "--device",
            str(args.device),
        ]
        evaluator_started = True
        subprocess.run(argv, cwd=REPO_ROOT, check=True)
        evaluator_outputs = _validate_evaluator_outputs(
            output_dir=output_dir,
            test_cache_path=test_cache_path,
            master=master,
        )
        promotion_receipt = {
            "schema_version": EVAL_RECEIPT_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "status": "complete",
            "completed_utc": _utc_now(),
            "global_eval_claim_path": str(claim_path),
            "master_selection_receipt_path": str(master_receipt_path),
            "master_selection_receipt_sha256": master["sha256"],
            "master_winner_run": master["winner_run"],
            "master_winner_family": master["family"],
            "selection_lock_path": artifacts["selection_lock_path"],
            "selection_lock_sha256": artifacts[
                "selection_lock_sha256"
            ],
            "checkpoint_path": artifacts["checkpoint_path"],
            "checkpoint_sha256": artifacts["checkpoint_sha256"],
            "locked_threshold": artifacts["lock"]["locked_threshold"],
            "encoder_fingerprint": artifacts["encoder_fingerprint"],
            "merge_receipt_path": str(merge_receipt_path),
            "merge_receipt_sha256": merge["sha256"],
            "test_cache_path": str(test_cache_path),
            "test_cache_sha256": cache["sha256"],
            "test_rows": cache["rows"],
            "evaluator": str(master["evaluator"]),
            "evaluator_sha256": dispatch_claim["evaluator_sha256"],
            **evaluator_outputs,
            "sealed_test_read": True,
            "sealed_test_evaluations": 1,
            "threshold_search_performed": False,
        }
        sealed._exclusive_json(
            promotion_receipt_path,
            promotion_receipt,
        )
        promotion_receipt_sha = _sha256(promotion_receipt_path)
        descriptor = os.open(
            promotion_sidecar_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                f"{promotion_receipt_sha}  "
                f"{promotion_receipt_path.name}\n"
            )
        sealed.atomic_json(
            claim_path,
            {
                **dispatch_claim,
                "status": "complete",
                "completed_utc": _utc_now(),
                "sealed_test_evaluations": 1,
                "promotion_receipt_path": str(promotion_receipt_path),
                "promotion_receipt_sha256": promotion_receipt_sha,
                "result_path": evaluator_outputs["result_path"],
                "result_sha256": evaluator_outputs["result_sha256"],
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "winner": master["winner_run"],
                    "family": master["family"],
                    "locked_threshold": artifacts["lock"][
                        "locked_threshold"
                    ],
                    "test_cache_sha256": cache["sha256"],
                    "result": evaluator_outputs["result_path"],
                    "result_sha256": evaluator_outputs["result_sha256"],
                    "promotion_receipt": str(promotion_receipt_path),
                    "promotion_receipt_sha256": promotion_receipt_sha,
                    "sealed_test_evaluations": 1,
                    "threshold_search_performed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        sealed.atomic_json(
            claim_path,
            {
                **claim,
                "status": "failed_after_global_eval_claim",
                "failed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_read": cache_read,
                "family_evaluator_started": evaluator_started,
                "sealed_test_evaluations": 0,
                "threshold_search_performed": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-selection-receipt", required=True)
    parser.add_argument("--merge-receipt", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sealed-test",
        action="store_true",
        help="explicit authorization for the sole sealed evaluation",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    command_evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
