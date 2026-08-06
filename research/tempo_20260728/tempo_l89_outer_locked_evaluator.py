#!/usr/bin/env python3
"""Composable exactly-once outer evaluator for the frozen L89 TEMPO family.

This file has three deliberately separated entry points:

``build-lock``
    Hash all development-selected artifacts and freeze components, weights,
    threshold, metric policy, and optional component interfaces.  The default
    product is a non-authorizable development template.
``dry-run``
    Replay the locked graph on the safe event-disjoint development caches.
``evaluate-once``
    Touch explicitly marked outer caches only after a final lock SHA and a
    literal one-time authorization token are both supplied.

This evaluator never extracts imagery itself.  A separately authorization-
gated producer can be cryptographically bound into the final lock; the
evaluator still accepts only its resulting aligned caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)
from research.tempo_20260728 import tempo_l89_global as global_runner  # noqa: E402


SCRIPT_VERSION = "tempo-l89-outer-locked-evaluator-v1"
FINALIZE_TOKEN = "FREEZE_FINAL_L89_OUTER_CANDIDATE"
EXECUTE_TOKEN = "RUN_EXACTLY_ONCE_LOCKED_L89_OUTER_EVALUATION"
HELD_OUT_RE = re.compile(
    r"(^|[._-])(test|sealed|holdout|outer)([._-]|$)",
    re.IGNORECASE,
)

RESEARCH_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727"
)
TEMPO_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_tempo_20260728/l89_global_v1"
)
PATCH_ROOT = Path(
    "/diniuvol/yuyao/methanefuse_research_20260728/"
    "tempo_l89_patch_p0_dim128_v1"
)
DEFAULT_DECISION_AUDIT = (
    PATCH_ROOT
    / "audits/scalar_multiseed_fixed_v1/fixed_multiseed_audit.json"
)
DEFAULT_BASE_DEV = RESEARCH_ROOT / "cache/l89_ragged_cls_v1/val.pt"
DEFAULT_P5_DEV = (
    RESEARCH_ROOT
    / "rctp_l89_sidecar_fallback_v1/cache/p5/val.pt"
)
DEFAULT_HEAD_CONFIG = (
    RESEARCH_ROOT
    / "rctp_l89_sidecar_fallback_v1/"
    "downstream_event_balanced_seed20260728/run_config.json"
)
DEFAULT_P0 = (
    RESEARCH_ROOT
    / "rctp_l89_sidecar_fallback_v1/"
    "downstream_event_balanced_seed20260728/p0/"
    "checkpoint_best_event_balanced_ap.pt"
)
DEFAULT_P5 = (
    RESEARCH_ROOT
    / "rctp_l89_sidecar_fallback_v1/"
    "downstream_event_balanced_seed20260728/p5/"
    "checkpoint_best_event_balanced_ap.pt"
)
DEFAULT_D1_CONFIG = (
    TEMPO_ROOT / "eventbase_d1_multiseed_fixed/run_config.json"
)
DEFAULT_D1 = tuple(
    TEMPO_ROOT
    / f"eventbase_d1_multiseed_fixed/seed_{seed}/"
    "d1_gated_delta_best_event_ap.pt"
    for seed in (20260727, 20260728, 20260729)
)
DEFAULT_TEMPLATE_DIR = (
    REPO_ROOT / "research/tempo_20260728/l89_lock_template_v4"
)
DEFAULT_PRODUCER_SPEC = (
    REPO_ROOT / "research/tempo_20260728/l89_producer_spec_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def development_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    offending = [
        part for part in resolved.parts if HELD_OUT_RE.search(part)
    ]
    if offending:
        raise ValueError(
            f"{purpose} contains held-out marker {sorted(offending)}: "
            f"{resolved}"
        )
    return resolved


def require_outer_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    if not any(HELD_OUT_RE.search(part) for part in resolved.parts):
        raise ValueError(
            f"{purpose} must be explicitly marked outer/test/sealed/holdout: "
            f"{resolved}"
        )
    return resolved


def artifact(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def normalize_weights(
    weights: Mapping[str, Any], *, expected: Iterable[str]
) -> dict[str, float]:
    expected_set = set(expected)
    observed = set(weights)
    if observed != expected_set:
        raise ValueError(
            f"Fusion weight keys {sorted(observed)} != "
            f"{sorted(expected_set)}"
        )
    output = {key: float(weights[key]) for key in sorted(weights)}
    if any(not math.isfinite(value) or value < 0 for value in output.values()):
        raise ValueError("Fusion weights must be finite and nonnegative.")
    if not math.isclose(sum(output.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Fusion weights must sum exactly to one.")
    return output


def fuse_locked_logits(
    logits: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    normalized = normalize_weights(weights, expected=logits)
    shapes = {tuple(np.asarray(value).shape) for value in logits.values()}
    if len(shapes) != 1:
        raise ValueError(f"Component logit shapes differ: {shapes}")
    output = np.zeros(next(iter(shapes)), dtype=np.float64)
    for name, weight in normalized.items():
        values = np.asarray(logits[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Component {name} contains non-finite logits.")
        output += weight * values
    return output


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _locked_patch_component(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load an optional, already-finalized patch logit-packet contract."""

    spec = read_json(path)
    required = {
        "schema_version",
        "name",
        "checkpoint_paths",
        "logit_columns",
        "within_family_weights",
        "fusion_weights",
        "selected_threshold",
        "expected_development_metrics",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Patch component spec is missing {missing}.")
    if spec["name"] != "patch":
        raise ValueError("Optional component must be named 'patch'.")
    columns = [str(value) for value in spec["logit_columns"]]
    family_weights = [float(value) for value in spec["within_family_weights"]]
    checkpoints = [Path(value).expanduser().resolve() for value in spec["checkpoint_paths"]]
    if not columns or len(columns) != len(family_weights):
        raise ValueError("Patch logit columns/weights differ in length.")
    if len(checkpoints) != len(columns):
        raise ValueError("Patch checkpoints/logit columns differ in length.")
    if not math.isclose(sum(family_weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Patch within-family weights must sum to one.")
    locked_artifacts = [artifact(path, role="optional patch component spec")]
    locked_artifacts.extend(
        artifact(checkpoint, role=f"patch checkpoint {index}")
        for index, checkpoint in enumerate(checkpoints)
    )
    component = {
        "adapter": "locked_logit_packet_v1",
        "name": "patch",
        "logit_columns": columns,
        "within_family_weights": family_weights,
        "checkpoint_artifacts": locked_artifacts[1:],
        "producer_contract": spec.get(
            "producer_contract",
            "must be frozen before final authorization",
        ),
    }
    return (
        {
            "component": component,
            "fusion_weights": normalize_weights(
                spec["fusion_weights"], expected=("p5", "d1", "patch")
            ),
            "selected_threshold": float(spec["selected_threshold"]),
            "expected_development_metrics": dict(
                spec["expected_development_metrics"]
            ),
        },
        locked_artifacts,
    )


def _locked_outer_producer(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind an opaque-target producer spec without touching its target CSV."""

    spec = read_json(path)
    if spec.get("schema_version") != "tempo-l89-outer-producer-spec-v1":
        raise ValueError("Unknown outer producer spec schema.")
    if spec.get("status") != "producer_template_locked_no_outer_read":
        raise RuntimeError("Outer producer spec has an unsafe status.")
    if bool(spec.get("test_or_sealed_or_holdout_read")):
        raise RuntimeError("Outer producer spec reports held-out access.")
    protocol = spec.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Outer producer spec has no protocol mapping.")
    if canonical_digest(protocol) != spec.get("protocol_sha256"):
        raise RuntimeError("Outer producer protocol digest changed.")
    if bool(protocol.get("target_csv_access_during_spec_build")):
        raise RuntimeError("Outer producer spec reports target access.")
    target_literal = protocol.get("target_csv_literal")
    if not isinstance(target_literal, str) or not target_literal:
        raise ValueError("Outer producer target literal is missing.")
    command_template = protocol.get("command_template")
    if not isinstance(command_template, Mapping):
        raise ValueError("Outer producer command template is missing.")
    target_digest = canonical_digest(target_literal)
    if command_template.get("outer_csv_literal_sha256") != target_digest:
        raise RuntimeError("Outer producer target literal digest changed.")
    producer_artifacts = protocol.get("locked_artifacts")
    if not isinstance(producer_artifacts, list) or not producer_artifacts:
        raise ValueError("Outer producer artifacts are missing.")
    # These are all safe development/code artifacts.  The opaque target
    # literal above is deliberately never converted to a Path.
    for record in producer_artifacts:
        if record.get("role") == "authorized outer producer":
            # The implementation filename itself contains the word "outer";
            # it is source code, not data.
            artifact_path = Path(record["path"]).expanduser().resolve()
        else:
            artifact_path = development_path(
                Path(record["path"]),
                purpose="outer producer locked artifact",
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if int(artifact_path.stat().st_size) != int(record["bytes"]):
            raise RuntimeError(
                f"Outer producer artifact size changed: {artifact_path}"
            )
        if sha256_file(artifact_path) != record["sha256"]:
            raise RuntimeError(
                f"Outer producer artifact digest changed: {artifact_path}"
            )
    spec_artifact = artifact(path, role="outer producer protocol spec")
    record = {
        "spec_artifact": spec_artifact,
        "protocol_sha256": str(spec["protocol_sha256"]),
        "target_csv_literal_sha256": target_digest,
        "command_template": dict(command_template),
        "extraction": dict(protocol["extraction"]),
        "output_contract": dict(protocol["output_contract"]),
    }
    return record, [spec_artifact]


def command_build_lock(args: argparse.Namespace) -> None:
    output_dir = development_path(
        Path(args.output_dir), purpose="lock output"
    )
    paths = {
        "decision_audit": development_path(
            Path(args.decision_audit), purpose="decision audit"
        ),
        "head_config": development_path(
            Path(args.head_config), purpose="head config"
        ),
        "p0_checkpoint": development_path(
            Path(args.p0_checkpoint), purpose="P0 checkpoint"
        ),
        "p5_checkpoint": development_path(
            Path(args.p5_checkpoint), purpose="P5 checkpoint"
        ),
        "d1_config": development_path(
            Path(args.d1_config), purpose="D1 config"
        ),
        "dev_base_cache": development_path(
            Path(args.dev_base_cache), purpose="base development cache"
        ),
        "dev_p5_cache": development_path(
            Path(args.dev_p5_cache), purpose="P5 development cache"
        ),
    }
    d1_paths = [
        development_path(Path(value), purpose="D1 checkpoint")
        for value in args.d1_checkpoint
    ]
    if len(d1_paths) != 3:
        raise ValueError("The frozen global candidate requires three D1 seeds.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing lock output: {output_dir}")
    decision = read_json(paths["decision_audit"])
    if bool(decision.get("test_or_sealed_read")):
        raise RuntimeError("Decision audit reports held-out access.")
    metrics = decision["metrics"]["p5_d1_fixed_equal_logit"]
    expected_seeds = [20260727, 20260728, 20260729]
    d1_artifacts: list[dict[str, Any]] = []
    observed_seeds: list[int] = []
    for checkpoint_path in d1_paths:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload.get("arm") != "d1_gated_delta":
            raise ValueError(f"Not a D1 checkpoint: {checkpoint_path}")
        observed_seeds.append(int(payload["seed"]))
        d1_artifacts.append(
            artifact(
                checkpoint_path,
                role=f"D1 seed {int(payload['seed'])} checkpoint",
            )
        )
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"D1 seed order {observed_seeds} != {expected_seeds}"
        )

    locked_artifacts = [
        artifact(paths["decision_audit"], role="development decision audit"),
        artifact(paths["head_config"], role="P0/P5 head run config"),
        artifact(paths["p0_checkpoint"], role="frozen P0 checkpoint"),
        artifact(paths["p5_checkpoint"], role="frozen P5 checkpoint"),
        artifact(paths["d1_config"], role="D1 run config"),
        *d1_artifacts,
        artifact(paths["dev_base_cache"], role="base development cache"),
        artifact(paths["dev_p5_cache"], role="P5 development cache"),
        artifact(
            Path(global_runner.__file__).resolve(),
            role="global model implementation",
        ),
        artifact(
            Path(cache_runner.__file__).resolve(),
            role="cache and head support implementation",
        ),
        artifact(Path(__file__).resolve(), role="locked evaluator source"),
    ]
    optional: Optional[dict[str, Any]] = None
    fusion_weights = {"d1": 0.5, "p5": 0.5}
    threshold = float(metrics["selected_threshold"])
    expected_metrics = dict(metrics)
    if args.patch_component_spec:
        patch_path = development_path(
            Path(args.patch_component_spec), purpose="patch component spec"
        )
        optional, patch_artifacts = _locked_patch_component(patch_path)
        locked_artifacts.extend(patch_artifacts)
        fusion_weights = optional["fusion_weights"]
        threshold = float(optional["selected_threshold"])
        expected_metrics = dict(optional["expected_development_metrics"])

    producer: Optional[dict[str, Any]] = None
    if args.outer_producer_spec:
        producer_path = development_path(
            Path(args.outer_producer_spec), purpose="outer producer spec"
        )
        producer, producer_artifacts = _locked_outer_producer(producer_path)
        locked_artifacts.extend(producer_artifacts)

    status = "development_template_not_authorizable"
    if args.lock_status == "final":
        if args.finalize_token != FINALIZE_TOKEN:
            raise RuntimeError("Final-lock confirmation token is missing.")
        if producer is None:
            raise RuntimeError(
                "A final lock requires an exact outer producer spec."
            )
        status = "locked_waiting_for_explicit_outer_authorization"
    protocol = {
        "schema_version": "tempo-l89-outer-protocol-v1",
        "evaluator_script_version": SCRIPT_VERSION,
        "candidate_id": (
            "p5_plus_d1_plus_patch_fixed"
            if optional is not None
            else "p5_plus_d1_fixed"
        ),
        "components": {
            "p5": {
                "adapter": "builtin_event_balanced_p5_v1",
                "checkpoint": next(
                    item
                    for item in locked_artifacts
                    if item["role"] == "frozen P5 checkpoint"
                ),
                "head_config": next(
                    item
                    for item in locked_artifacts
                    if item["role"] == "P0/P5 head run config"
                ),
            },
            "d1": {
                "adapter": "builtin_l89_d1_v1",
                "arm": "d1_gated_delta",
                "seeds": expected_seeds,
                "within_family_logit_weights": [1.0 / 3.0] * 3,
                "checkpoints": d1_artifacts,
                "p0_checkpoint": next(
                    item
                    for item in locked_artifacts
                    if item["role"] == "frozen P0 checkpoint"
                ),
                "d1_config": next(
                    item
                    for item in locked_artifacts
                    if item["role"] == "D1 run config"
                ),
            },
        },
        "optional_component": (
            None if optional is None else optional["component"]
        ),
        "fusion": {
            "operator": "fixed_weight_logit_sum",
            "weights": fusion_weights,
            "selected_threshold": threshold,
            "threshold_comparator": ">=",
            "weight_search_on_outer": False,
            "threshold_search_on_outer": False,
        },
        "expected_development_metrics": expected_metrics,
        "metric_policy": {
            "primary": (
                "event_balanced_positive_f1_selected",
                "event_balanced_macro_f1_selected",
            ),
            "ranking": ("event_balanced_ap", "event_balanced_auc"),
            "canonical_event_equal_total_weight": True,
            "selection_performed_on_outer": False,
        },
        "outer_input": {
            "implemented_mode": (
                "authorization_gated_bound_producer_then_aligned_cache_pair"
                if producer is not None
                else "precomputed_aligned_cache_pair_only"
            ),
            "base_cache_shape": "[N,6,768]",
            "p5_cache_shape": "[N,6,1536]",
            "p5_first_768_must_equal_base": True,
            "raw_csv_extraction": (
                "implemented only by the separately authorized bound producer"
                if producer is not None
                else "disabled because no producer is bound"
            ),
            "source_passes_per_cache": 1,
            "producer": producer,
        },
        "prohibitions": [
            "no outer threshold fitting",
            "no outer checkpoint or epoch selection",
            "no outer ensemble-weight fitting",
            "no outer calibration",
            "no subgroup selection",
            "no repeat under the same output intent",
            "no raw CSV input in evaluator v1",
        ],
        "locked_artifacts": locked_artifacts,
    }
    manifest = {
        "schema_version": "tempo-l89-outer-lock-manifest-v1",
        "status": status,
        "protocol": protocol,
        "protocol_sha256": canonical_digest(protocol),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_or_sealed_or_holdout_read": False,
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "LOCK_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    atomic_text(
        output_dir / "LOCK_MANIFEST.sha256",
        f"{manifest_sha}  {manifest_path.name}\n",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "protocol_sha256": manifest["protocol_sha256"],
                "status": status,
                "candidate_id": protocol["candidate_id"],
                "test_or_sealed_or_holdout_read": False,
            },
            indent=2,
        ),
        flush=True,
    )


def validate_lock(
    path: Path, *, require_final: bool
) -> tuple[dict[str, Any], str]:
    manifest = read_json(path)
    manifest_sha = sha256_file(path)
    allowed = {
        "development_template_not_authorizable",
        "locked_waiting_for_explicit_outer_authorization",
    }
    if manifest.get("status") not in allowed:
        raise RuntimeError("Unknown or mutable lock status.")
    if require_final and (
        manifest["status"]
        != "locked_waiting_for_explicit_outer_authorization"
    ):
        raise RuntimeError("A development template cannot read outer data.")
    if bool(manifest.get("test_or_sealed_or_holdout_read")):
        raise RuntimeError("Lock construction reports held-out access.")
    protocol = manifest["protocol"]
    if canonical_digest(protocol) != manifest["protocol_sha256"]:
        raise RuntimeError("Protocol digest changed.")
    for record in protocol["locked_artifacts"]:
        artifact_path = Path(record["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Locked artifact disappeared: {artifact_path}")
        if int(artifact_path.stat().st_size) != int(record["bytes"]):
            raise RuntimeError(f"Locked artifact size changed: {artifact_path}")
        if sha256_file(artifact_path) != record["sha256"]:
            raise RuntimeError(f"Locked artifact digest changed: {artifact_path}")
    component_names = {"p5", "d1"}
    if protocol["optional_component"] is not None:
        component_names.add("patch")
    normalize_weights(
        protocol["fusion"]["weights"], expected=component_names
    )
    if protocol["fusion"]["weight_search_on_outer"]:
        raise RuntimeError("Outer weight search was enabled.")
    if protocol["fusion"]["threshold_search_on_outer"]:
        raise RuntimeError("Outer threshold search was enabled.")
    return manifest, manifest_sha


def prepare_cache(
    payload: Mapping[str, Any],
    *,
    source_name: str,
    expected_dim: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source_name} is not a cache mapping.")
    split = str(payload.get("split", ""))
    if not split:
        raise ValueError(f"{source_name} has no split marker.")
    cache_runner.validate_cache_payload(
        payload, path=Path(source_name), expected_split=split
    )
    if tuple(payload["features"].shape[1:]) != (6, expected_dim):
        raise ValueError(
            f"{source_name} feature shape {tuple(payload['features'].shape)} "
            f"does not end in (6,{expected_dim})."
        )
    rows = global_runner.take_development_rows(payload)
    rows["role_index"] = payload["role_index"].long().clone()
    rows["role_names"] = tuple(str(value) for value in payload["role_names"])
    rows["t0_index"] = int(payload["t0_index"])
    rows["split"] = split
    return rows


def audit_cache_alignment(
    base: Mapping[str, Any], p5: Mapping[str, Any]
) -> dict[str, Any]:
    if len(base["ids"]) != len(p5["ids"]):
        raise ValueError("Base/P5 usable row counts differ.")
    for key in ("ids", "plume_ids", "event_ids"):
        if list(base[key]) != list(p5[key]):
            raise ValueError(f"Base/P5 ordered {key} differ.")
    if not torch.equal(base["labels"].long(), p5["labels"].long()):
        raise ValueError("Base/P5 labels differ.")
    for key in ("unique_mask", "delta_days", "valid_fraction", "role_index"):
        left = base[key]
        right = p5[key].to(dtype=left.dtype)
        if not torch.equal(left, right):
            raise ValueError(f"Base/P5 {key} differ.")
    if base["role_names"] != p5["role_names"]:
        raise ValueError("Base/P5 role names differ.")
    if base["t0_index"] != 0 or p5["t0_index"] != 0:
        raise ValueError("Locked candidate requires t0 at role index zero.")
    for indices in torch.arange(len(base["labels"])).split(512):
        if not torch.equal(
            base["features"][indices],
            p5["features"][indices, :, :768],
        ):
            raise ValueError("P5 first 768 feature dimensions differ from base.")
    return {
        "rows": int(len(base["labels"])),
        "events": int(len(set(base["event_ids"]))),
        "ordered_id_plume_event_label_exact": True,
        "mask_gap_quality_exact": True,
        "p5_first_768_exact": True,
        "role_names": list(base["role_names"]),
    }


def build_role_head(
    checkpoint: Mapping[str, Any],
    head_config: Mapping[str, Any],
    *,
    expected_arm: str,
) -> global_runner.FrozenRoleOnlyBase:
    if checkpoint.get("arm") != expected_arm:
        raise ValueError(
            f"Expected {expected_arm} checkpoint, got {checkpoint.get('arm')}"
        )
    cache_audit = checkpoint["cache_audit"]
    periods = tuple(
        float(value)
        for value in str(
            head_config.get("delta_periods", "1,3,7,30,90,365")
        ).split(",")
    )
    head = cache_runner.RaggedCurrentQueryHead(
        feature_dim=int(cache_audit["feature_dim"]),
        num_roles=int(cache_audit["timepoints"]),
        model_dim=int(head_config["model_dim"]),
        num_heads=int(head_config["num_heads"]),
        depth=2,
        mlp_ratio=float(head_config["mlp_ratio"]),
        dropout=float(head_config["dropout"]),
        periods_days=periods,
        t0_index=int(cache_audit["t0_index"]),
    )
    head.load_state_dict(checkpoint["model"], strict=True)
    return global_runner.FrozenRoleOnlyBase(head).eval()


def infer_role_logits(
    model: global_runner.FrozenRoleOnlyBase,
    rows: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device).eval()
    output = torch.empty(len(rows["labels"]), dtype=torch.float32)
    with torch.inference_mode():
        for indices in torch.arange(len(output)).split(int(batch_size)):
            output[indices] = model(
                rows["features"][indices].to(device),
                rows["unique_mask"][indices].to(device),
                rows["delta_days"][indices].to(device),
                rows["role_index"].to(device),
            ).cpu()
    return output.numpy().astype(np.float64)


def build_d1_model(
    checkpoint: Mapping[str, Any],
    *,
    p0_checkpoint: Mapping[str, Any],
    head_config: Mapping[str, Any],
    d1_config: Mapping[str, Any],
) -> global_runner.TEMPOGlobalResidual:
    if checkpoint.get("arm") != "d1_gated_delta":
        raise ValueError("Expected a D1 checkpoint.")
    base = build_role_head(
        p0_checkpoint, head_config, expected_arm="p0"
    )
    periods = tuple(
        float(value)
        for value in str(d1_config["delta_periods"]).split(",")
    )
    model = global_runner.TEMPOGlobalResidual(
        base,
        feature_dim=int(d1_config["cache_audit"]["feature_dim"]),
        num_roles=int(d1_config["cache_audit"]["timepoints"]),
        t0_index=int(d1_config["t0_index"]),
        temporal_dim=int(d1_config["temporal_dim"]),
        dropout=float(d1_config["dropout"]),
        periods_days=periods,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def infer_d1_logits(
    model: global_runner.TEMPOGlobalResidual,
    rows: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device).eval()
    output = torch.empty(len(rows["labels"]), dtype=torch.float32)
    with torch.inference_mode():
        for indices in torch.arange(len(output)).split(int(batch_size)):
            output[indices] = model(
                rows["features"][indices].to(device),
                rows["unique_mask"][indices].to(device),
                rows["delta_days"][indices].to(device),
                rows["valid_fraction"][indices].to(device),
                rows["role_index"].to(device),
                arm="d1_gated_delta",
            ).cpu()
    return output.numpy().astype(np.float64)


def patch_logits_from_packet(
    frame: pd.DataFrame,
    *,
    rows: Mapping[str, Any],
    component: Mapping[str, Any],
) -> np.ndarray:
    required = {"id", "event_id", "label", *component["logit_columns"]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Patch logit packet is missing {missing}.")
    if frame["id"].astype(str).tolist() != list(rows["ids"]):
        raise ValueError("Patch packet ordered IDs differ.")
    if frame["event_id"].astype(str).tolist() != list(rows["event_ids"]):
        raise ValueError("Patch packet ordered event IDs differ.")
    if not np.array_equal(
        frame["label"].to_numpy(dtype=np.int64),
        rows["labels"].numpy().astype(np.int64),
    ):
        raise ValueError("Patch packet labels differ.")
    matrix = np.stack(
        [
            frame[column].to_numpy(dtype=np.float64)
            for column in component["logit_columns"]
        ],
        axis=0,
    )
    weights = np.asarray(
        component["within_family_weights"], dtype=np.float64
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Patch packet contains non-finite logits.")
    return np.average(matrix, axis=0, weights=weights)


def evaluate_payloads(
    *,
    base_payload: Mapping[str, Any],
    p5_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    patch_packet: Optional[pd.DataFrame],
    batch_size: int,
    torch_threads: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    base_rows = prepare_cache(
        base_payload, source_name="base_cache", expected_dim=768
    )
    p5_rows = prepare_cache(
        p5_payload, source_name="p5_cache", expected_dim=1536
    )
    alignment = audit_cache_alignment(base_rows, p5_rows)
    protocol = manifest["protocol"]
    artifacts = {
        item["role"]: Path(item["path"])
        for item in protocol["locked_artifacts"]
    }
    head_config = read_json(artifacts["P0/P5 head run config"])
    d1_config = read_json(artifacts["D1 run config"])
    p0_checkpoint = torch.load(
        artifacts["frozen P0 checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    p5_checkpoint = torch.load(
        artifacts["frozen P5 checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("cpu")
    p5_model = build_role_head(
        p5_checkpoint, head_config, expected_arm="p5"
    )
    p5_logit = infer_role_logits(
        p5_model,
        p5_rows,
        batch_size=batch_size,
        device=device,
    )
    del p5_model
    d1_logits: list[np.ndarray] = []
    d1_records = protocol["components"]["d1"]["checkpoints"]
    for record in d1_records:
        checkpoint = torch.load(
            record["path"], map_location="cpu", weights_only=False
        )
        model = build_d1_model(
            checkpoint,
            p0_checkpoint=p0_checkpoint,
            head_config=head_config,
            d1_config=d1_config,
        )
        d1_logits.append(
            infer_d1_logits(
                model,
                base_rows,
                batch_size=batch_size,
                device=device,
            )
        )
        del model
    d1_logit = np.average(
        np.stack(d1_logits, axis=0),
        axis=0,
        weights=np.asarray(
            protocol["components"]["d1"]["within_family_logit_weights"],
            dtype=np.float64,
        ),
    )
    component_logits: dict[str, np.ndarray] = {
        "p5": p5_logit,
        "d1": d1_logit,
    }
    optional = protocol["optional_component"]
    if optional is None:
        if patch_packet is not None:
            raise ValueError("Patch packet supplied but patch is not locked.")
    else:
        if patch_packet is None:
            raise ValueError("Locked patch component requires a logit packet.")
        component_logits["patch"] = patch_logits_from_packet(
            patch_packet, rows=base_rows, component=optional
        )
    final_logit = fuse_locked_logits(
        component_logits, protocol["fusion"]["weights"]
    )
    probability = stable_sigmoid(final_logit)
    threshold = float(protocol["fusion"]["selected_threshold"])
    labels = base_rows["labels"].numpy().astype(np.int64)
    metrics = global_runner.metric_bundle(
        labels,
        probability,
        base_rows["event_ids"],
        threshold=threshold,
    )
    output = pd.DataFrame(
        {
            "id": base_rows["ids"],
            "plume_id": base_rows["plume_ids"],
            "event_id": base_rows["event_ids"],
            "label": labels,
            "p5_logit": p5_logit,
            "d1_ensemble_logit": d1_logit,
            "final_logit": final_logit,
            "probability": probability,
            "prediction": (probability >= threshold).astype(np.int8),
        }
    )
    if "patch" in component_logits:
        output["patch_ensemble_logit"] = component_logits["patch"]
    result = {
        "schema_version": "tempo-l89-outer-result-v1",
        "candidate_id": protocol["candidate_id"],
        "metrics": metrics,
        "threshold": threshold,
        "threshold_refit": False,
        "fusion_weights": dict(protocol["fusion"]["weights"]),
        "d1_seeds": list(protocol["components"]["d1"]["seeds"]),
        "d1_within_family_logit_weights": list(
            protocol["components"]["d1"]["within_family_logit_weights"]
        ),
        "cache_alignment": alignment,
        "selection_performed": False,
    }
    return result, output


def regression_errors(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, float]:
    keys = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_macro_f1_selected",
        "event_balanced_positive_f1_selected",
        "row_ap",
        "row_auc",
        "all_negative_fp_mass",
    )
    return {
        key: float(abs(float(observed[key]) - float(expected[key])))
        for key in keys
    }


def write_outputs(
    *,
    output_dir: Path,
    result: Mapping[str, Any],
    predictions: pd.DataFrame,
    receipt: dict[str, Any],
) -> None:
    metrics_path = output_dir / "LOCKED_METRICS.json"
    predictions_path = output_dir / "LOCKED_PREDICTIONS.csv"
    atomic_json(metrics_path, result)
    predictions.to_csv(predictions_path, index=False)
    receipt["outputs"] = {
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    receipt["status"] = "complete"
    receipt["completed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    atomic_json(output_dir / "RECEIPT.json", receipt)


def command_dry_run(args: argparse.Namespace) -> None:
    manifest_path = development_path(
        Path(args.lock_manifest), purpose="lock manifest"
    )
    base_path = development_path(
        Path(args.base_cache), purpose="base development cache"
    )
    p5_path = development_path(
        Path(args.p5_cache), purpose="P5 development cache"
    )
    output_dir = development_path(
        Path(args.output_dir), purpose="dry-run output"
    )
    patch_path = (
        None
        if args.patch_packet is None
        else development_path(
            Path(args.patch_packet), purpose="patch development packet"
        )
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing dry-run output: {output_dir}")
    manifest, manifest_sha = validate_lock(
        manifest_path, require_final=False
    )
    base_payload = torch.load(
        base_path, map_location="cpu", weights_only=False
    )
    p5_payload = torch.load(
        p5_path, map_location="cpu", weights_only=False
    )
    patch_packet = (
        None if patch_path is None else pd.read_csv(patch_path)
    )
    result, predictions = evaluate_payloads(
        base_payload=base_payload,
        p5_payload=p5_payload,
        manifest=manifest,
        patch_packet=patch_packet,
        batch_size=int(args.batch_size),
        torch_threads=int(args.torch_threads),
    )
    errors = regression_errors(
        result["metrics"],
        manifest["protocol"]["expected_development_metrics"],
    )
    tolerance = float(args.regression_tolerance)
    if max(errors.values(), default=0.0) > tolerance:
        raise RuntimeError(
            f"Development replay exceeded tolerance {tolerance}: {errors}"
        )
    output_dir.mkdir(parents=True)
    result["development_regression"] = {
        "absolute_errors": errors,
        "tolerance": tolerance,
        "passed": True,
    }
    receipt = {
        "schema_version": "tempo-l89-outer-receipt-v1",
        "mode": "development_dry_run",
        "status": "running",
        "lock_manifest": str(manifest_path),
        "lock_manifest_sha256": manifest_sha,
        "inputs": {
            "base_cache": str(base_path),
            "base_cache_sha256": sha256_file(base_path),
            "p5_cache": str(p5_path),
            "p5_cache_sha256": sha256_file(p5_path),
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "host": platform.node(),
        },
        "test_or_sealed_or_holdout_read": False,
        "selection_performed": False,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_outputs(
        output_dir=output_dir,
        result=result,
        predictions=predictions,
        receipt=receipt,
    )
    print(
        json.dumps(
            {
                "candidate_id": result["candidate_id"],
                "metrics": result["metrics"],
                "development_regression": result["development_regression"],
            },
            indent=2,
        ),
        flush=True,
    )


def exclusive_intent(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage_once(
    source: Path, destination: Path, *, role: str
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Refusing existing staged source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with source.open("rb", buffering=0) as input_stream:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
        )
        with os.fdopen(descriptor, "wb", buffering=0) as output_stream:
            while True:
                block = input_stream.read(16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                output_stream.write(block)
                total += len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    return {
        "role": role,
        "source_path": str(source),
        "staged_path": str(destination),
        "bytes_read": int(total),
        "sha256_during_single_pass": digest.hexdigest(),
        "source_content_passes": 1,
    }


def command_evaluate_once(args: argparse.Namespace) -> None:
    # Authorization is checked before any outer source path is stat'ed/opened.
    if args.confirm != EXECUTE_TOKEN:
        raise RuntimeError("Exactly-once L89 authorization token is missing.")
    manifest_path = Path(args.lock_manifest).expanduser().resolve()
    manifest, manifest_sha = validate_lock(
        manifest_path, require_final=True
    )
    if args.authorized_lock_sha256 != manifest_sha:
        raise RuntimeError("Authorized lock SHA does not match the manifest.")
    base_source = require_outer_path(
        Path(args.base_cache), purpose="outer base cache"
    )
    p5_source = require_outer_path(
        Path(args.p5_cache), purpose="outer P5 cache"
    )
    patch_source = (
        None
        if args.patch_packet is None
        else require_outer_path(
            Path(args.patch_packet), purpose="outer patch packet"
        )
    )
    if manifest["protocol"]["optional_component"] is None:
        if patch_source is not None:
            raise ValueError("Patch packet supplied to a non-patch lock.")
    elif patch_source is None:
        raise ValueError("Final patch lock requires its outer logit packet.")
    if not base_source.is_file() or not p5_source.is_file():
        raise FileNotFoundError("One or both locked outer caches are missing.")
    if patch_source is not None and not patch_source.is_file():
        raise FileNotFoundError(patch_source)
    output_dir = Path(args.output_dir).expanduser().resolve()
    staging_dir = Path(args.staging_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Exactly-once output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    started = pd.Timestamp.now(tz="UTC").isoformat()
    exclusive_intent(
        output_dir / "EXACTLY_ONCE_INTENT.json",
        {
            "status": "outer_source_read_committed",
            "lock_manifest": str(manifest_path),
            "lock_manifest_sha256": manifest_sha,
            "source_paths": [
                str(base_source),
                str(p5_source),
                *([] if patch_source is None else [str(patch_source)]),
            ],
            "started_utc": started,
            "rerun_allowed": False,
        },
    )
    prefix = manifest_sha[:16]
    staged_base = staging_dir / f"{prefix}_base_{base_source.name}"
    staged_p5 = staging_dir / f"{prefix}_p5_{p5_source.name}"
    stage_receipts = [
        stage_once(base_source, staged_base, role="base_cache"),
        stage_once(p5_source, staged_p5, role="p5_cache"),
    ]
    staged_patch: Optional[Path] = None
    if patch_source is not None:
        staged_patch = staging_dir / f"{prefix}_patch_{patch_source.name}"
        stage_receipts.append(
            stage_once(
                patch_source, staged_patch, role="patch_logit_packet"
            )
        )
    # The remote/source files are never opened again.
    base_payload = torch.load(
        staged_base, map_location="cpu", weights_only=False
    )
    p5_payload = torch.load(
        staged_p5, map_location="cpu", weights_only=False
    )
    patch_packet = (
        None if staged_patch is None else pd.read_csv(staged_patch)
    )
    result, predictions = evaluate_payloads(
        base_payload=base_payload,
        p5_payload=p5_payload,
        manifest=manifest,
        patch_packet=patch_packet,
        batch_size=int(args.batch_size),
        torch_threads=int(args.torch_threads),
    )
    result["evaluation_scope"] = "outer_exactly_once"
    receipt = {
        "schema_version": "tempo-l89-outer-receipt-v1",
        "mode": "outer_exactly_once",
        "status": "running",
        "lock_manifest": str(manifest_path),
        "lock_manifest_sha256": manifest_sha,
        "input_single_pass_receipts": stage_receipts,
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "host": platform.node(),
        },
        "test_or_sealed_or_holdout_read": True,
        "selection_performed": False,
        "rerun_allowed": False,
        "started_utc": started,
    }
    write_outputs(
        output_dir=output_dir,
        result=result,
        predictions=predictions,
        receipt=receipt,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_id": result["candidate_id"],
                "lock_manifest_sha256": manifest_sha,
                "receipt": str(output_dir / "RECEIPT.json"),
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock = subparsers.add_parser(
        "build-lock", help="Build a template or explicitly finalized lock."
    )
    lock.set_defaults(function=command_build_lock)
    lock.add_argument("--decision-audit", default=str(DEFAULT_DECISION_AUDIT))
    lock.add_argument("--head-config", default=str(DEFAULT_HEAD_CONFIG))
    lock.add_argument("--p0-checkpoint", default=str(DEFAULT_P0))
    lock.add_argument("--p5-checkpoint", default=str(DEFAULT_P5))
    lock.add_argument("--d1-config", default=str(DEFAULT_D1_CONFIG))
    lock.add_argument(
        "--d1-checkpoint",
        action="append",
        default=None,
        help="Repeat exactly three times in frozen seed order.",
    )
    lock.add_argument("--dev-base-cache", default=str(DEFAULT_BASE_DEV))
    lock.add_argument("--dev-p5-cache", default=str(DEFAULT_P5_DEV))
    lock.add_argument("--patch-component-spec")
    lock.add_argument("--outer-producer-spec")
    lock.add_argument(
        "--lock-status", choices=("template", "final"), default="template"
    )
    lock.add_argument("--finalize-token")
    lock.add_argument("--output-dir", default=str(DEFAULT_TEMPLATE_DIR))

    dry = subparsers.add_parser(
        "dry-run", help="Replay the locked graph on safe development caches."
    )
    dry.set_defaults(function=command_dry_run)
    dry.add_argument("--lock-manifest", required=True)
    dry.add_argument("--base-cache", default=str(DEFAULT_BASE_DEV))
    dry.add_argument("--p5-cache", default=str(DEFAULT_P5_DEV))
    dry.add_argument("--patch-packet")
    dry.add_argument("--output-dir", required=True)
    dry.add_argument("--batch-size", type=int, default=512)
    dry.add_argument("--torch-threads", type=int, default=16)
    # CPU/GPU float32 replay differs by a few ulps.  On rows exactly beside
    # the frozen threshold this can move weighted F1 by about 8.5e-5 even
    # though AP/AUC differ by less than 4e-7.
    dry.add_argument("--regression-tolerance", type=float, default=1e-4)

    outer = subparsers.add_parser(
        "evaluate-once",
        help="Run one explicitly authorized outer evaluation.",
    )
    outer.set_defaults(function=command_evaluate_once)
    outer.add_argument("--lock-manifest", required=True)
    outer.add_argument("--base-cache", required=True)
    outer.add_argument("--p5-cache", required=True)
    outer.add_argument("--patch-packet")
    outer.add_argument("--staging-dir", required=True)
    outer.add_argument("--output-dir", required=True)
    outer.add_argument("--authorized-lock-sha256", required=True)
    outer.add_argument("--confirm", required=True)
    outer.add_argument("--batch-size", type=int, default=512)
    outer.add_argument("--torch-threads", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-lock" and args.d1_checkpoint is None:
        args.d1_checkpoint = [str(path) for path in DEFAULT_D1]
    args.function(args)


if __name__ == "__main__":
    main()
