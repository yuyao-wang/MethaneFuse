#!/usr/bin/env python3
"""Derive a strict S2-only cache from a full legacy360 feature cache.

The full cache stores three per-visit CLS tokens for every sensor as well as
the exact three-visit concatenated-band checkpoint logit.  This utility keeps
only rows for which all three S2 visits and the verified S2 checkpoint head
are available, then removes the other sensor axes.  It rejects sealed-test
caches during model selection; a test cache can be converted only when an
already-written S2 selection lock and its checkpoint SHA validate first.

The resulting cache remains compatible with
``query360_two_axis_full_legacy.py train`` using ``--base-mode hybrid`` and
either ``--arm current_only`` or ``--arm two_axis_query``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


CACHE_SCHEMA = "query360-two-axis-feature-cache-v1"
ALLOWED_SPLITS = {"train_core", "train", "dev", "evaluation"}
ROW_TENSOR_KEYS = {
    "features",
    "features_hybrid",
    "valid_mask",
    "base_sensor_logits",
    "base_sensor_valid",
    "base_sensor_logits_hybrid",
    "base_sensor_valid_hybrid",
    "base_fused_logits",
    "base_hybrid_logits",
    "labels",
}
ROW_LIST_KEYS = {
    "ids",
    "plume_ids",
    "event_ids",
    "availability_signatures",
}


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_cache(
    payload: Mapping[str, Any],
    source: Path,
    *,
    expected_split: Optional[str] = None,
    allow_sealed_test: bool = False,
) -> int:
    if payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"{source}: unsupported cache schema")
    split = str(payload.get("split"))
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"{source}: split={split!r}, expected={expected_split!r}"
        )
    is_sealed_test = split == "test" or bool(payload.get("sealed_test_read"))
    if is_sealed_test and not allow_sealed_test:
        raise ValueError(f"{source}: sealed-test cache is forbidden")
    if not is_sealed_test and split not in ALLOWED_SPLITS:
        raise ValueError(
            f"{source}: split={split!r}; only train_core/dev caches are allowed"
        )
    if allow_sealed_test and (
        split != "test" or not bool(payload.get("sealed_test_read"))
    ):
        raise ValueError(
            f"{source}: authorized sealed conversion requires a test cache "
            "marked sealed_test_read=true"
        )
    features = payload.get("features_hybrid", payload.get("features"))
    valid = payload.get("valid_mask")
    labels = payload.get("labels")
    if not torch.is_tensor(features) or features.ndim != 4:
        raise ValueError(f"{source}: malformed features")
    if not torch.is_tensor(valid) or valid.shape != features.shape[:3]:
        raise ValueError(f"{source}: malformed valid_mask")
    if valid.dtype != torch.bool:
        raise ValueError(f"{source}: valid_mask must be boolean")
    if not torch.is_tensor(labels) or labels.shape != features.shape[:1]:
        raise ValueError(f"{source}: malformed labels")
    sensor_names = list(payload.get("sensor_names", []))
    if "s2" not in sensor_names:
        raise ValueError(f"{source}: cache has no S2 axis")
    if len(sensor_names) != features.shape[1]:
        raise ValueError(f"{source}: sensor_names/features disagree")
    return sensor_names.index("s2")


def make_s2_only_payload(
    source_payload: Mapping[str, Any],
    *,
    source_path: Path,
    expected_split: Optional[str] = None,
    allow_sealed_test: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an S2-only payload and a compact audit without reading images."""

    s2_index = _require_source_cache(
        source_payload,
        source_path,
        expected_split=expected_split,
        allow_sealed_test=allow_sealed_test,
    )
    is_sealed_test = str(source_payload.get("split")) == "test"
    features = source_payload.get(
        "features_hybrid", source_payload["features"]
    )
    valid = source_payload["valid_mask"]
    hybrid_sensor_valid = source_payload.get(
        "base_sensor_valid_hybrid",
        source_payload.get("base_sensor_valid"),
    )
    hybrid_sensor_logits = source_payload.get(
        "base_sensor_logits_hybrid",
        source_payload.get("base_sensor_logits"),
    )
    hybrid_fused_logits = source_payload.get(
        "base_hybrid_logits",
        source_payload.get("base_fused_logits"),
    )
    if not torch.is_tensor(hybrid_sensor_valid):
        raise ValueError(f"{source_path}: no hybrid sensor-valid tensor")
    if not torch.is_tensor(hybrid_sensor_logits):
        raise ValueError(f"{source_path}: no hybrid sensor-logit tensor")
    if not torch.is_tensor(hybrid_fused_logits):
        raise ValueError(f"{source_path}: no hybrid fused-logit tensor")
    if hybrid_sensor_valid.shape != valid.shape[:2]:
        raise ValueError(f"{source_path}: malformed hybrid sensor-valid tensor")
    if hybrid_sensor_logits.shape != valid.shape[:2]:
        raise ValueError(f"{source_path}: malformed hybrid sensor-logit tensor")
    if hybrid_fused_logits.shape != valid.shape[:1]:
        raise ValueError(f"{source_path}: malformed hybrid fused-logit tensor")

    complete_three_visit = valid[:, s2_index, :].all(dim=1)
    exact_concat_base = hybrid_sensor_valid[:, s2_index]
    selected = complete_three_visit & exact_concat_base
    positions = torch.nonzero(selected, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError(f"{source_path}: no exact three-visit S2 rows")

    # For selected rows, the full hybrid definition must have chosen the
    # verified S2 checkpoint logit rather than the universal fallback.
    expected_base = hybrid_sensor_logits[positions, s2_index].float()
    observed_base = hybrid_fused_logits[positions].float()
    max_base_difference = float(
        (expected_base - observed_base).abs().max().item()
    )
    if max_base_difference > 1e-6:
        raise RuntimeError(
            f"{source_path}: hybrid base is not the exact S2 head "
            f"(max_abs_difference={max_base_difference})"
        )

    output = copy.copy(dict(source_payload))
    # Universal row fusion may have used additional sensors, so it is not a
    # valid S2-only base and is removed to make --base-mode universal fail
    # closed.  Hybrid is the only supported base for this derived cache.
    for key in (
        "features_universal",
        "base_universal_logits",
        "base_sensor_logits_universal",
        "base_sensor_valid_universal",
    ):
        output.pop(key, None)

    for key in ROW_TENSOR_KEYS:
        value = source_payload.get(key)
        if not torch.is_tensor(value):
            continue
        if value.shape[0] != valid.shape[0]:
            raise ValueError(f"{source_path}: {key} is not row aligned")
        if key in {
            "features",
            "features_hybrid",
            "valid_mask",
        }:
            output[key] = value[positions, s2_index : s2_index + 1].clone()
        elif key in {
            "base_sensor_logits",
            "base_sensor_valid",
            "base_sensor_logits_hybrid",
            "base_sensor_valid_hybrid",
        }:
            output[key] = value[
                positions, s2_index : s2_index + 1
            ].clone()
        else:
            output[key] = value[positions].clone()

    # Normalize backward-compatible aliases so every training path sees the
    # same verified S2 checkpoint base and S2-backbone per-visit features.
    output["features"] = output["features_hybrid"]
    output["base_sensor_logits"] = output["base_sensor_logits_hybrid"]
    output["base_sensor_valid"] = output["base_sensor_valid_hybrid"]
    output["base_fused_logits"] = output["base_hybrid_logits"]

    for key in ROW_LIST_KEYS:
        values = source_payload.get(key)
        if values is None:
            continue
        if len(values) != valid.shape[0]:
            raise ValueError(f"{source_path}: {key} is not row aligned")
        if key == "availability_signatures":
            output[key] = ["s2"] * int(positions.numel())
        else:
            output[key] = [values[index] for index in positions.tolist()]

    output["sensor_names"] = ["s2"]
    output["sealed_test_read"] = is_sealed_test
    output["base_definitions"] = {
        "hybrid": (
            "exact verified S2 checkpoint logit from the original "
            "three-visit concatenated-band input"
        )
    }
    source_manifest = dict(source_payload.get("manifest", {}))
    output["manifest"] = {
        **source_manifest,
        "rows": int(positions.numel()),
        "source_rows": int(valid.shape[0]),
        "derived_subset": "s2_exact_three_visit",
    }
    output["s2_subset"] = {
        "source_cache": str(source_path.expanduser().absolute()),
        "source_rows": int(valid.shape[0]),
        "selected_rows": int(positions.numel()),
        "selection": (
            "all three S2 per-visit tokens valid AND verified concatenated-"
            "band S2 checkpoint logit available"
        ),
        "other_sensor_axes_removed": True,
        "supported_base_mode": "hybrid",
        "supported_arms": ["current_only", "two_axis_query"],
        "base_semantics": (
            "Both arms are residual adapters on the same strong temporal "
            "three-visit S2 checkpoint. current_only changes only the new "
            "residual context; it is not a from-scratch t0-only baseline."
        ),
    }

    output_valid = output["valid_mask"]
    if output_valid.shape[1:] != (1, valid.shape[2]):
        raise AssertionError("S2 sensor axis was not reduced correctly")
    if not output_valid.all():
        raise AssertionError("derived cache contains an invalid S2 visit")
    if not output["base_sensor_valid_hybrid"].all():
        raise AssertionError("derived cache contains a missing S2 base logit")
    if not torch.equal(output["labels"], source_payload["labels"][positions]):
        raise AssertionError("labels changed while subsetting")
    if not torch.isfinite(output["features"].float()).all():
        raise AssertionError("derived S2 features are non-finite")
    if not torch.isfinite(output["base_hybrid_logits"].float()).all():
        raise AssertionError("derived S2 base logits are non-finite")

    labels = output["labels"]
    audit = {
        "schema_version": "legacy360-s2-feature-subset-audit-v1",
        "source_cache": str(source_path.expanduser().absolute()),
        "split": str(output["split"]),
        "source_rows": int(valid.shape[0]),
        "complete_three_visit_rows": int(complete_three_visit.sum()),
        "exact_concat_base_rows": int(exact_concat_base.sum()),
        "selected_rows": int(positions.numel()),
        "dropped_rows": int(valid.shape[0] - positions.numel()),
        "feature_shape": list(output["features"].shape),
        "max_hybrid_vs_s2_base_logit_abs_difference": max_base_difference,
        "sealed_test_read": is_sealed_test,
    }
    audit["labels"] = (
        "withheld until locked evaluation"
        if is_sealed_test
        else {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        }
    )
    return output, audit


def _validate_selection_lock(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        lock = json.load(stream)
    if lock.get("schema_version") != "query360-two-axis-selection-lock-v1":
        raise ValueError(f"{path}: unsupported selection-lock schema")
    if lock.get("base_mode") != "hybrid":
        raise ValueError(f"{path}: S2-only cache requires base_mode=hybrid")
    if lock.get("arm") not in {"current_only", "two_axis_query"}:
        raise ValueError(f"{path}: unsupported S2 arm {lock.get('arm')!r}")
    model_config = lock.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError(f"{path}: selection lock has no model_config")
    if int(model_config.get("num_sensors", -1)) != 1:
        raise ValueError(f"{path}: selection lock is not an S2-only model")
    checkpoint = Path(str(lock.get("checkpoint", ""))).expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{path}: locked checkpoint is missing")
    if _sha256_file(checkpoint) != lock.get("checkpoint_sha256"):
        raise ValueError(f"{path}: locked checkpoint SHA mismatch")
    if bool(lock.get("test_cache_read_before_lock")):
        raise ValueError(f"{path}: lock reports premature test-cache access")
    return lock


def command_subset(args: argparse.Namespace) -> None:
    source = Path(args.input_cache).expanduser().absolute()
    destination = Path(args.output_cache).expanduser().absolute()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite")
    lock: Optional[dict[str, Any]] = None
    if args.expected_split == "test":
        if not args.sealed_test:
            raise PermissionError(
                "refusing sealed-test conversion without --sealed-test"
            )
        if not args.selection_lock:
            raise ValueError(
                "--selection-lock is required before sealed-test conversion"
            )
        # Validate the immutable development selection before touching the
        # test-cache path.
        lock = _validate_selection_lock(
            Path(args.selection_lock).expanduser().absolute()
        )
    elif args.sealed_test or args.selection_lock:
        raise ValueError(
            "--sealed-test/--selection-lock are valid only for "
            "--expected-split test"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    output, audit = make_s2_only_payload(
        payload,
        source_path=source,
        expected_split=args.expected_split,
        allow_sealed_test=args.expected_split == "test",
    )
    if lock is not None:
        lock_path = Path(args.selection_lock).expanduser().absolute()
        lock_record = {
            "path": str(lock_path),
            "sha256": _sha256_file(lock_path),
            "checkpoint_sha256": lock["checkpoint_sha256"],
            "best_epoch": int(lock["best_epoch"]),
            "arm": lock["arm"],
            "locked_threshold": float(lock["locked_threshold"]),
        }
        output["s2_subset"]["selection_lock_validated_before_source_read"] = (
            lock_record
        )
        audit["selection_lock_validated_before_source_read"] = lock_record
    _atomic_torch(destination, output)
    audit_path = destination.with_suffix(destination.suffix + ".audit.json")
    audit["output_cache"] = str(destination)
    _atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument(
        "--expected-split",
        choices=["train_core", "train", "dev", "evaluation", "test"],
        required=True,
    )
    parser.add_argument("--selection-lock", default="")
    parser.add_argument(
        "--sealed-test",
        action="store_true",
        help=(
            "permit test-cache conversion only after validating a completed "
            "S2-only development selection lock"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    command_subset(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
