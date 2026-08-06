#!/usr/bin/env python3
"""Authorization-gated producer for official L89 per-time outer caches.

The producer reuses the exact audited development implementation:

1. stream the authorized source CSV once into local staging while hashing;
2. extract six independent Panopticon CLS tokens per row;
3. derive P5's response-sidecar features from that base cache without a second
   image pass.

``build-spec`` never resolves, stats, lists, hashes, or opens the target outer
CSV.  ``produce-once`` checks the final evaluation lock SHA and literal
authorization token before the first filesystem operation on that CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as base_runner,
)
from research.pretraining_20260727 import (  # noqa: E402
    rctp_l89_sidecar_fallback as sidecar_runner,
)
from research.tempo_20260728 import (  # noqa: E402
    tempo_l89_outer_locked_evaluator as evaluator,
)


SCRIPT_VERSION = "tempo-l89-outer-cache-producer-v1"
PRODUCE_TOKEN = "PRODUCE_EXACTLY_ONCE_LOCKED_L89_OUTER_CACHES"

# This is an opaque string until produce-once passes authorization.  Never
# construct a Path from it during build-spec.
DEFAULT_TARGET_CSV_LITERAL = (
    "/home/yuyao/panopticon/Upgraded_dataset/"
    "l89_6time_temporal_hard_event_filtered_split/"
    "L89_temporal_test_hard_event_filtered.csv"
)
DEFAULT_DEV_CACHE = (
    Path("/diniuvol/yuyao/methanefuse_research_20260727")
    / "cache/l89_ragged_cls_v1/val.pt"
)
DEFAULT_WEIGHTS = (
    REPO_ROOT / "weights/panopticon_vitb14_teacher.pth"
)
DEFAULT_P5_SIDECAR = (
    Path("/diniuvol/yuyao/methanefuse_research_20260727")
    / "rctp_l89_sidecar_fallback_v1/pretrain/p5/"
    "sidecar_best_dev_ap.pt"
)
DEFAULT_SPEC = (
    REPO_ROOT
    / "research/tempo_20260728/l89_producer_spec_v1.json"
)


def build_spec(args: argparse.Namespace) -> None:
    output_path = evaluator.development_path(
        Path(args.output), purpose="producer spec output"
    )
    dev_cache_path = evaluator.development_path(
        Path(args.dev_cache), purpose="development reference cache"
    )
    weights_path = evaluator.development_path(
        Path(args.weights), purpose="Panopticon weights"
    )
    sidecar_path = evaluator.development_path(
        Path(args.p5_sidecar_checkpoint), purpose="P5 sidecar checkpoint"
    )
    if output_path.exists():
        raise FileExistsError(output_path)
    for path in (dev_cache_path, weights_path, sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Only safe development artifacts are opened here.
    reference = torch.load(
        dev_cache_path, map_location="cpu", weights_only=False
    )
    if not isinstance(reference, Mapping):
        raise ValueError("Development reference is not a cache mapping.")
    base_runner.validate_cache_payload(
        reference,
        path=dev_cache_path,
        expected_split=str(reference["split"]),
    )
    if tuple(reference["features"].shape[1:]) != (6, 768):
        raise ValueError("Reference is not the audited six-role 768-D cache.")
    contract = dict(reference["input_contract"])
    weights_sha = evaluator.sha256_file(weights_path)
    if weights_sha != str(reference["weights_sha256"]):
        raise RuntimeError("Reference cache and declared weights differ.")
    sidecar_checkpoint = torch.load(
        sidecar_path, map_location="cpu", weights_only=False
    )
    if sidecar_checkpoint.get("arm") != "p5_correct_response":
        raise ValueError("Declared P5 sidecar checkpoint has the wrong arm.")
    if sidecar_checkpoint.get("base_weights_sha256") != weights_sha:
        raise RuntimeError("P5 sidecar and base Panopticon weights differ.")
    sidecar_config = dict(sidecar_checkpoint["sidecar_config"])
    expected_sidecar = {
        "feature_dim": 768,
        "response_dim": 16,
        "rank": 8,
        "residual_scale": 0.1,
    }
    for key, expected in expected_sidecar.items():
        if float(sidecar_config[key]) != float(expected):
            raise RuntimeError(
                f"P5 sidecar {key}={sidecar_config[key]} != {expected}"
            )
    artifacts = [
        evaluator.artifact(
            Path(__file__).resolve(), role="authorized outer producer"
        ),
        evaluator.artifact(
            Path(base_runner.__file__).resolve(),
            role="per-time CLS extractor implementation",
        ),
        evaluator.artifact(
            Path(sidecar_runner.__file__).resolve(),
            role="P5 sidecar cache implementation",
        ),
        evaluator.artifact(
            weights_path, role="frozen Panopticon weights"
        ),
        evaluator.artifact(
            sidecar_path, role="frozen P5 response-sidecar checkpoint"
        ),
        evaluator.artifact(
            dev_cache_path, role="safe development input-contract reference"
        ),
    ]
    extraction = {
        "semantic_split": "outer",
        "cache_split_value": "outer",
        "path_columns": list(contract["path_columns"]),
        "time_columns": list(contract["time_columns"]),
        "role_names": list(contract["role_names"]),
        "label_column": "label",
        "id_column": "id",
        "plume_id_column": "plume_id",
        "event_column": "event_group_id",
        "band_indices": list(contract["band_indices"]),
        "channel_ids": list(contract["channel_ids"]),
        "normalization_mean": list(contract["normalization_mean"]),
        "normalization_std": list(contract["normalization_std"]),
        "normalization_source": str(contract["normalization_source"]),
        "image_size": int(contract["image_size"]),
        "min_valid_fraction": float(contract["min_valid_fraction"]),
        "validity_band_index": int(contract["validity_band_index"]),
        "zero_invalid_pixels": bool(contract["zero_invalid_pixels"]),
        "duplicate_rule": str(contract["duplicate_rule"]),
        "max_rows": 0,
        "row_selection_seed": int(contract["row_selection_seed"]),
        "max_invalid_t0": 0,
        "max_read_errors": 0,
        "amp_dtype": "bfloat16",
        "storage_dtype": "float16",
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "persistent_workers": True,
        "local_cache_mode": "sync",
        "local_cache_workers": int(args.local_cache_workers),
        "local_cache_min_free_gb": float(args.local_cache_min_free_gb),
        "local_cache_bypass_root": "/diniuvol/yuyao",
        "sidecar_batch_size": 512,
        "base_output_name": "base_cache.pt",
        "p5_output_name": "p5_cache.pt",
        "staged_csv_name": "source_manifest.csv",
    }
    command_template = {
        "entrypoint": str(Path(__file__).resolve()),
        "subcommand": "produce-once",
        "required_runtime_arguments": [
            "--lock-manifest",
            "--authorized-lock-sha256",
            "--confirm",
            "--producer-spec",
            "--outer-csv",
            "--output-dir",
            "--local-image-cache-dir",
            "--device",
        ],
        "outer_csv_literal_sha256": evaluator.canonical_digest(
            str(args.target_csv_literal)
        ),
        "fusion_evaluator_input_outputs": [
            extraction["base_output_name"],
            extraction["p5_output_name"],
        ],
    }
    protocol = {
        "schema_version": "tempo-l89-outer-producer-protocol-v1",
        "producer_script_version": SCRIPT_VERSION,
        "target_csv_literal": str(args.target_csv_literal),
        "target_csv_access_during_spec_build": False,
        "source_csv_policy": (
            "after lock+token only: one sequential source pass into local "
            "staging while hashing; all later CSV reads use staged copy"
        ),
        "image_policy": (
            "one base Panopticon extraction; P5 is derived from cached base "
            "CLS and never rereads imagery"
        ),
        "extraction": extraction,
        "p5_sidecar_contract": {
            "arm": "p5",
            "checkpoint_arm": str(sidecar_checkpoint["arm"]),
            "objective_response": [
                float(value)
                for value in sidecar_checkpoint["objective_response"]
            ],
            "sidecar_config": sidecar_config,
            "base_channel_byte_identical": True,
            "residual_scale": 0.1,
        },
        "command_template": command_template,
        "locked_artifacts": artifacts,
        "output_contract": {
            "base_shape": "[N,6,768]",
            "p5_shape": "[N,6,1536]",
            "ordered_id_plume_event_label_exact": True,
            "mask_gap_quality_exact": True,
            "p5_first_768_byte_identical_to_base": True,
            "cache_sha256_bound_in_producer_receipt": True,
            "overwrite": False,
            "rerun_allowed": False,
        },
    }
    spec = {
        "schema_version": "tempo-l89-outer-producer-spec-v1",
        "status": "producer_template_locked_no_outer_read",
        "protocol": protocol,
        "protocol_sha256": evaluator.canonical_digest(protocol),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_or_sealed_or_holdout_read": False,
    }
    evaluator.atomic_json(output_path, spec)
    digest = evaluator.sha256_file(output_path)
    evaluator.atomic_text(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{digest}  {output_path.name}\n",
    )
    print(
        json.dumps(
            {
                "producer_spec": str(output_path),
                "producer_spec_sha256": digest,
                "protocol_sha256": spec["protocol_sha256"],
                "target_csv_access_during_spec_build": False,
                "status": spec["status"],
            },
            indent=2,
        ),
        flush=True,
    )


def validate_spec(path: Path) -> tuple[dict[str, Any], str]:
    spec = evaluator.read_json(path)
    digest = evaluator.sha256_file(path)
    if spec.get("status") != "producer_template_locked_no_outer_read":
        raise RuntimeError("Producer spec status changed.")
    if bool(spec.get("test_or_sealed_or_holdout_read")):
        raise RuntimeError("Producer spec reports held-out access.")
    protocol = spec["protocol"]
    if evaluator.canonical_digest(protocol) != spec["protocol_sha256"]:
        raise RuntimeError("Producer protocol digest changed.")
    for record in protocol["locked_artifacts"]:
        artifact_path = Path(record["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if int(artifact_path.stat().st_size) != int(record["bytes"]):
            raise RuntimeError(f"Producer artifact size changed: {artifact_path}")
        if evaluator.sha256_file(artifact_path) != record["sha256"]:
            raise RuntimeError(
                f"Producer artifact digest changed: {artifact_path}"
            )
    return spec, digest


def _producer_record_from_lock(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    producer = manifest["protocol"]["outer_input"].get("producer")
    if not isinstance(producer, Mapping):
        raise RuntimeError("Final evaluation lock has no outer producer.")
    return producer


def _cache_namespace(
    *,
    spec: Mapping[str, Any],
    staged_csv: Path,
    base_output: Path,
    local_image_cache_dir: Path,
    device: str,
) -> argparse.Namespace:
    extraction = spec["protocol"]["extraction"]
    artifacts = {
        item["role"]: item
        for item in spec["protocol"]["locked_artifacts"]
    }
    return argparse.Namespace(
        csv=str(staged_csv),
        split=str(extraction["cache_split_value"]),
        output_cache=str(base_output),
        weights=artifacts["frozen Panopticon weights"]["path"],
        path_columns=",".join(extraction["path_columns"]),
        time_columns=",".join(extraction["time_columns"]),
        label_column=str(extraction["label_column"]),
        id_column=str(extraction["id_column"]),
        plume_id_column=str(extraction["plume_id_column"]),
        event_column=str(extraction["event_column"]),
        band_indices=",".join(
            str(value) for value in extraction["band_indices"]
        ),
        stats_json=None,
        image_size=int(extraction["image_size"]),
        min_valid_fraction=float(extraction["min_valid_fraction"]),
        validity_band_index=int(extraction["validity_band_index"]),
        zero_invalid_pixels=bool(extraction["zero_invalid_pixels"]),
        batch_size=int(extraction["batch_size"]),
        num_workers=int(extraction["num_workers"]),
        prefetch_factor=int(extraction["prefetch_factor"]),
        persistent_workers=bool(extraction["persistent_workers"]),
        device=str(device),
        amp_dtype=str(extraction["amp_dtype"]),
        storage_dtype=str(extraction["storage_dtype"]),
        local_cache_dir=str(local_image_cache_dir),
        local_cache_bypass_root=str(
            extraction["local_cache_bypass_root"]
        ),
        local_cache_mode=str(extraction["local_cache_mode"]),
        local_cache_workers=int(extraction["local_cache_workers"]),
        local_cache_min_free_gb=float(
            extraction["local_cache_min_free_gb"]
        ),
        max_rows=int(extraction["max_rows"]),
        row_selection_seed=int(extraction["row_selection_seed"]),
        max_invalid_t0=int(extraction["max_invalid_t0"]),
        max_read_errors=int(extraction["max_read_errors"]),
        log_interval=20,
        overwrite=False,
        debug=False,
    )


def produce_once(args: argparse.Namespace) -> None:
    # Do not construct a Path from --outer-csv before these checks.
    if args.confirm != PRODUCE_TOKEN:
        raise RuntimeError("Exactly-once producer authorization token is missing.")
    manifest_path = Path(args.lock_manifest).expanduser().resolve()
    manifest, manifest_sha = evaluator.validate_lock(
        manifest_path, require_final=True
    )
    if args.authorized_lock_sha256 != manifest_sha:
        raise RuntimeError("Authorized lock SHA does not match.")
    producer_record = _producer_record_from_lock(manifest)
    spec_path = evaluator.development_path(
        Path(args.producer_spec), purpose="producer spec"
    )
    spec, spec_sha = validate_spec(spec_path)
    locked_spec = producer_record["spec_artifact"]
    if (
        str(spec_path) != str(Path(locked_spec["path"]).resolve())
        or spec_sha != locked_spec["sha256"]
    ):
        raise RuntimeError("Producer spec is not the one in the final lock.")
    target_literal = str(spec["protocol"]["target_csv_literal"])
    if str(args.outer_csv) != target_literal:
        raise RuntimeError("Outer CSV literal differs from the locked target.")

    output_dir = evaluator.require_outer_path(
        Path(args.output_dir), purpose="outer producer output directory"
    )
    local_image_cache_dir = Path(
        args.local_image_cache_dir
    ).expanduser().resolve()

    # First filesystem interaction with the outer CSV occurs only here.
    outer_csv = evaluator.require_outer_path(
        Path(args.outer_csv), purpose="official outer CSV"
    )
    if not outer_csv.is_file():
        raise FileNotFoundError(outer_csv)
    if output_dir.exists():
        raise FileExistsError(
            f"Exactly-once producer output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    started = pd.Timestamp.now(tz="UTC").isoformat()
    evaluator.exclusive_intent(
        output_dir / "EXACTLY_ONCE_PRODUCER_INTENT.json",
        {
            "status": "outer_csv_read_committed",
            "lock_manifest": str(manifest_path),
            "lock_manifest_sha256": manifest_sha,
            "producer_spec": str(spec_path),
            "producer_spec_sha256": spec_sha,
            "outer_csv_literal": target_literal,
            "started_utc": started,
            "rerun_allowed": False,
        },
    )
    extraction = spec["protocol"]["extraction"]
    staged_csv = output_dir / str(extraction["staged_csv_name"])
    source_receipt = evaluator.stage_once(
        outer_csv, staged_csv, role="official_outer_csv"
    )
    base_output = output_dir / str(extraction["base_output_name"])
    p5_output = output_dir / str(extraction["p5_output_name"])
    artifacts = {
        item["role"]: item
        for item in spec["protocol"]["locked_artifacts"]
    }
    receipt: dict[str, Any] = {
        "schema_version": "tempo-l89-outer-producer-receipt-v1",
        "status": "running",
        "lock_manifest": str(manifest_path),
        "lock_manifest_sha256": manifest_sha,
        "producer_spec": str(spec_path),
        "producer_spec_sha256": spec_sha,
        "outer_csv_single_pass": source_receipt,
        "runtime": {
            "device": str(args.device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "host": platform.node(),
        },
        "started_utc": started,
        "test_or_sealed_or_holdout_read": True,
        "rerun_allowed": False,
    }
    try:
        base_runner.cache_features(
            _cache_namespace(
                spec=spec,
                staged_csv=staged_csv,
                base_output=base_output,
                local_image_cache_dir=local_image_cache_dir,
                device=str(args.device),
            )
        )
        sidecar_runner.command_build_cache(
            argparse.Namespace(
                arm="p5",
                split=str(extraction["cache_split_value"]),
                base_cache=str(base_output),
                sidecar_checkpoint=artifacts[
                    "frozen P5 response-sidecar checkpoint"
                ]["path"],
                output_cache=str(p5_output),
                batch_size=int(extraction["sidecar_batch_size"]),
            )
        )
        base_payload = torch.load(
            base_output, map_location="cpu", weights_only=False
        )
        p5_payload = torch.load(
            p5_output, map_location="cpu", weights_only=False
        )
        base_rows = evaluator.prepare_cache(
            base_payload,
            source_name="produced_base_cache",
            expected_dim=768,
        )
        p5_rows = evaluator.prepare_cache(
            p5_payload,
            source_name="produced_p5_cache",
            expected_dim=1536,
        )
        alignment = evaluator.audit_cache_alignment(base_rows, p5_rows)
        receipt["outputs"] = {
            "base_cache": str(base_output),
            "base_cache_sha256": evaluator.sha256_file(base_output),
            "base_feature_sha256": str(base_payload["feature_sha256"]),
            "p5_cache": str(p5_output),
            "p5_cache_sha256": evaluator.sha256_file(p5_output),
            "p5_feature_sha256": str(p5_payload["feature_sha256"]),
            "alignment": alignment,
        }
        receipt["status"] = "complete"
        receipt["completed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        evaluator.atomic_json(
            output_dir / "PRODUCER_RECEIPT.json", receipt
        )
    except Exception as error:
        receipt["status"] = "failed_no_rerun"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        receipt["failed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        with suppress(Exception):
            evaluator.atomic_json(
                output_dir / "PRODUCER_FAILURE.json", receipt
            )
        raise
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(output_dir / "PRODUCER_RECEIPT.json"),
                "base_cache": receipt["outputs"]["base_cache"],
                "p5_cache": receipt["outputs"]["p5_cache"],
                "outer_csv_source_passes": 1,
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    spec = subparsers.add_parser(
        "build-spec",
        help="Freeze producer code/config from safe development artifacts.",
    )
    spec.set_defaults(function=build_spec)
    spec.add_argument(
        "--target-csv-literal", default=DEFAULT_TARGET_CSV_LITERAL
    )
    spec.add_argument("--dev-cache", default=str(DEFAULT_DEV_CACHE))
    spec.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    spec.add_argument(
        "--p5-sidecar-checkpoint", default=str(DEFAULT_P5_SIDECAR)
    )
    spec.add_argument("--batch-size", type=int, default=16)
    spec.add_argument("--num-workers", type=int, default=8)
    spec.add_argument("--prefetch-factor", type=int, default=2)
    spec.add_argument("--local-cache-workers", type=int, default=8)
    spec.add_argument("--local-cache-min-free-gb", type=float, default=20.0)
    spec.add_argument("--output", default=str(DEFAULT_SPEC))

    produce = subparsers.add_parser(
        "produce-once",
        help="After explicit authorization, produce outer base/P5 caches once.",
    )
    produce.set_defaults(function=produce_once)
    produce.add_argument("--lock-manifest", required=True)
    produce.add_argument("--authorized-lock-sha256", required=True)
    produce.add_argument("--confirm", required=True)
    produce.add_argument("--producer-spec", required=True)
    produce.add_argument("--outer-csv", required=True)
    produce.add_argument("--output-dir", required=True)
    produce.add_argument("--local-image-cache-dir", required=True)
    produce.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
