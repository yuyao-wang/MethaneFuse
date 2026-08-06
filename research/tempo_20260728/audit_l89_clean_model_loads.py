#!/usr/bin/env python3
"""Audit every model checkpoint loaded by the clean-L89 formal pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache


FORBIDDEN_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout|outer)([._/\\-]|$)", re.IGNORECASE
)
SEEDS = (20260727, 20260728, 20260729)
FROZEN_BASE_SHA256 = (
    "55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e"
)
KNOWN_OLD_CHECKPOINT_SHA256 = {
    # Prior sidecars trained on the old lineage.
    "ba3f22aeeb5d9978419e4e0197476a4f62fbe76ca68bbe10c3bf5fd804cc5aff",
    "6f9ec7ce1c4a33c509f354ce9f1f817bfa2fbb1d4ec23879be96a56aff1502c3",
    # Prior event-balanced P0/P4/P5 heads.
    "e5924ecff3db6ed9fe99105d355e082229b3c8d3a15a15da715e2bf0d2a5fe56",
    "2268de582e181f987470cd720aae77cf009fa08e3c0ff398faafbb526260a3f3",
    "5a10899243183d64eebb939f50b3dc2b6a54364f781c7691efc91e2a5f46d70c",
    # Prior D1 three-seed checkpoints.
    "9da6acb5216139ccc68fe7eac65cd4074d0bb095629e651bb4333f230177c17a",
    "d8a0c149ff029797b389d1142161d1ba6bfd38590fae9b2daeb1a2e3f451d288",
    "72f82bb16c8495a95c8f6acf7e6c890f30702dfa8539b02732a15af6d1b78255",
}


def safe_path(value: str | Path, *, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    if FORBIDDEN_RE.search(str(path)):
        raise ValueError(f"{purpose} contains a held-out marker: {path}")
    return path


def require_within(path: Path, root: Path, *, purpose: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{purpose} escapes the new formal lineage: {path} not below {root}"
        ) from error


def artifact(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sha = cache.sha256_file(path)
    if sha in KNOWN_OLD_CHECKPOINT_SHA256:
        raise ValueError(
            f"{role} is byte-identical to a forbidden old checkpoint: {sha}"
        )
    return {
        "role": role,
        "path": str(path),
        "sha256": sha,
        "bytes": int(path.stat().st_size),
    }


def require_exact_audited_cache(
    audit: Mapping[str, Any],
    *,
    key: str,
    expected: Path,
    formal_root: Path,
    purpose: str,
) -> dict[str, str]:
    """Verify both the exact current-run cache path and its recorded SHA."""

    actual = safe_path(audit[key], purpose=purpose)
    require_within(actual, formal_root, purpose=purpose)
    if actual != expected.resolve():
        raise ValueError(
            f"{purpose} points to {actual}, expected exactly {expected.resolve()}."
        )
    sha_key = f"{key}_sha256"
    observed_sha = cache.sha256_file(actual)
    if audit.get(sha_key) != observed_sha:
        raise ValueError(f"{purpose} SHA mismatch.")
    return {"path": str(actual), "sha256": observed_sha}


def planned_paths(root: Path) -> dict[str, Any]:
    return {
        "fresh_base_caches": [
            str(root / "base_cache/train.pt"),
            str(root / "base_cache/val.pt"),
        ],
        "fresh_sidecars": [
            str(root / "sidecar/pretrain/p4/sidecar_best_dev_ap.pt"),
            str(root / "sidecar/pretrain/p5/sidecar_best_dev_ap.pt"),
        ],
        "fresh_heads": [
            str(
                root
                / f"event_balanced_heads_seed20260728/{arm}/"
                "checkpoint_best_event_balanced_ap.pt"
            )
            for arm in ("p0", "p4", "p5")
        ],
        "fresh_d1": [
            str(
                root
                / f"tempo_d1_three_seed/seed_{seed}/"
                "d1_gated_delta_best_event_ap.pt"
            )
            for seed in SEEDS
        ],
    }


def write_preload_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = safe_path(args.formal_root, purpose="formal root")
    weights = safe_path(args.base_weights, purpose="frozen base weights")
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weights_sha = cache.sha256_file(weights)
    if weights_sha != FROZEN_BASE_SHA256:
        raise ValueError(
            f"Frozen Panopticon SHA {weights_sha} != {FROZEN_BASE_SHA256}."
        )
    output = root / "MODEL_LOAD_LEDGER.json"
    if output.exists():
        raise FileExistsError(output)
    planned = planned_paths(root)
    for collection in planned.values():
        for value in collection:
            require_within(
                Path(value).resolve(), root, purpose="planned model artifact"
            )
    payload = {
        "schema_version": "l89-clean-model-load-ledger-v1",
        "status": "preload_plan_locked",
        "created_before_first_formal_model_load": True,
        "formal_root": str(root),
        "allowed_external_model_loads": [
            {
                "path": str(weights),
                "sha256": weights_sha,
                "reason": "frozen Panopticon initialization explicitly reused",
            }
        ],
        "planned_current_run_artifacts": planned,
        "resume_or_optimizer_state_load_allowed": False,
        "old_checkpoint_reuse_allowed": False,
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache.atomic_json_write(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = safe_path(args.formal_root, purpose="formal root")
    weights = safe_path(args.base_weights, purpose="frozen base weights")
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weights_sha = cache.sha256_file(weights)
    if weights_sha != FROZEN_BASE_SHA256:
        raise ValueError(
            f"Frozen Panopticon SHA {weights_sha} != {FROZEN_BASE_SHA256}."
        )
    preload_output = root / "MODEL_LOAD_LEDGER.json"
    if not preload_output.is_file():
        raise FileNotFoundError(
            "Pre-load model ledger was not created before training: "
            f"{preload_output}"
        )
    preload = json.loads(preload_output.read_text(encoding="utf-8"))
    if preload.get("status") != "preload_plan_locked":
        raise ValueError("Model ledger is not a locked pre-load plan.")
    if (
        preload.get("created_before_first_formal_model_load") is not True
        or preload.get("formal_root") != str(root)
        or preload.get("resume_or_optimizer_state_load_allowed") is not False
        or preload.get("old_checkpoint_reuse_allowed") is not False
        or preload.get("test_or_sealed_or_holdout_or_outer_read") is not False
        or preload.get("allowed_external_model_loads")
        != [
            {
                "path": str(weights),
                "sha256": weights_sha,
                "reason": "frozen Panopticon initialization explicitly reused",
            }
        ]
    ):
        raise ValueError("Model ledger pre-load contract changed.")
    if preload.get("planned_current_run_artifacts") != planned_paths(root):
        raise ValueError("Model ledger planned paths changed.")
    preload_sha = cache.sha256_file(preload_output)
    final_output = root / "MODEL_LOAD_LEDGER_FINAL.json"
    if final_output.exists():
        raise FileExistsError(final_output)

    base_caches = {
        "train": root / "base_cache/train.pt",
        "val": root / "base_cache/val.pt",
    }
    sidecars = {
        "p4": root / "sidecar/pretrain/p4/sidecar_best_dev_ap.pt",
        "p5": root / "sidecar/pretrain/p5/sidecar_best_dev_ap.pt",
    }
    heads = {
        arm: root
        / f"event_balanced_heads_seed20260728/{arm}/"
        "checkpoint_best_event_balanced_ap.pt"
        for arm in ("p0", "p4", "p5")
    }
    d1_checkpoints = {
        seed: root
        / f"tempo_d1_three_seed/seed_{seed}/"
        "d1_gated_delta_best_event_ap.pt"
        for seed in SEEDS
    }
    for path in (
        *base_caches.values(),
        *sidecars.values(),
        *heads.values(),
        *d1_checkpoints.values(),
    ):
        require_within(path.resolve(), root, purpose="formal artifact")

    loads: list[dict[str, Any]] = []
    loads.append(
        {
            **artifact(weights, role="frozen_panopticon_base"),
            "external_load_allowed": True,
            "consumers": [
                "base CLS extraction",
                "P4 sidecar pretext",
                "P5 sidecar pretext",
            ],
        }
    )
    for split, path in base_caches.items():
        payload = cache.torch_load_trusted(path)
        cache.validate_cache_payload(payload, path=path, expected_split=split)
        if payload["weights_sha256"] != weights_sha:
            raise ValueError(f"{split} base cache weights SHA mismatch.")
        loads.append(
            {
                **artifact(path, role=f"fresh_base_cache_{split}"),
                "generated_with_base_weights_sha256": weights_sha,
                "external_load_allowed": False,
            }
        )

    split_audit_path = root / "sidecar_split/AUDIT.json"
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    if split_audit.get("event_overlap") != 0:
        raise ValueError("Sidecar fit/capability audit event overlap is nonzero.")
    if split_audit.get(
        "formal_inner_development_used_for_sidecar_checkpoint_selection"
    ) is not False:
        raise ValueError("Formal inner development entered sidecar selection.")
    selection = split_audit["selection"]
    if int(selection["capability_eligible_negative_rows"]) < 512:
        raise ValueError("Sidecar capability has fewer than 512 eligible negatives.")
    if selection["locked_train_plan_lengths_by_epoch"] != [2048, 2048]:
        raise ValueError("Sidecar train plan lengths are not [2048, 2048].")
    if int(selection["locked_capability_plan_length"]) != 512:
        raise ValueError("Sidecar capability plan length is not 512.")
    source_cache = safe_path(
        split_audit["source"]["train_cache"],
        purpose="sidecar split source cache",
    )
    if source_cache != base_caches["train"].resolve():
        raise ValueError("Sidecar split did not derive from fresh base train.")
    if (
        split_audit["source"]["train_cache_sha256"]
        != cache.sha256_file(base_caches["train"])
    ):
        raise ValueError("Sidecar split source cache SHA mismatch.")
    if split_audit["source"].get(
        "ordered_id_plume_event_label_exact"
    ) is not True:
        raise ValueError("Sidecar source ordered identity audit is incomplete.")
    source_digests = split_audit["source"]["row_tensor_digests"]
    source_train_payload = cache.torch_load_trusted(base_caches["train"])
    for key in (
        "labels",
        "timestamps_utc_ns",
        "valid_mask",
        "unique_mask",
        "delta_days",
        "valid_fraction",
    ):
        observed = cache.tensor_sha256(source_train_payload[key])
        if source_digests.get(f"{key}_sha256") != observed:
            raise ValueError(f"Sidecar split source {key} digest mismatch.")

    expected_sidecar_arm = {
        "p4": "p4_response_scrambled",
        "p5": "p5_correct_response",
    }
    sidecar_configs: dict[str, Mapping[str, Any]] = {}
    sidecar_train_plan_shas: dict[str, list[str]] = {}
    for arm, path in sidecars.items():
        config_path = path.parent / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        sidecar_configs[arm] = config
        expected_config_paths = {
            "train_csv": root / "sidecar_split/train.csv",
            "dev_csv": root / "sidecar_split/capability.csv",
            "train_cache": root / "sidecar_split/train.pt",
            "dev_cache": root / "sidecar_split/capability.pt",
        }
        for key, expected in expected_config_paths.items():
            configured = safe_path(
                config["args"][key], purpose=f"{arm} sidecar {key}"
            )
            if configured != expected.resolve():
                raise ValueError(
                    f"{arm} sidecar {key} escaped the nested train-only split."
                )
        cache_audit = config["cache_audit"]
        if int(cache_audit["event_overlap"]) != 0:
            raise ValueError(f"{arm} sidecar cache event overlap is nonzero.")
        for key, expected in (
            ("train_cache", expected_config_paths["train_cache"]),
            ("validation_cache", expected_config_paths["dev_cache"]),
        ):
            require_exact_audited_cache(
                cache_audit,
                key=key,
                expected=expected,
                formal_root=root,
                purpose=f"{arm} sidecar audited {key}",
            )
        history_path = path.parent / "metrics_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) != 2 or [int(row["epoch"]) for row in history] != [1, 2]:
            raise ValueError(f"{arm} sidecar did not run exactly two epochs.")
        sidecar_train_plan_shas[arm] = [
            str(row["train_plan_sha256"]) for row in history
        ]
        payload = cache.torch_load_trusted(path)
        if payload.get("arm") != expected_sidecar_arm[arm]:
            raise ValueError(f"{arm} sidecar arm mismatch.")
        if payload.get("base_weights_sha256") != weights_sha:
            raise ValueError(f"{arm} sidecar base weights SHA mismatch.")
        best_ap = max(
            float(record["dev"]["average_precision"]) for record in history
        )
        expected_epoch = next(
            int(record["epoch"])
            for record in history
            if float(record["dev"]["average_precision"]) == best_ap
        )
        if (
            int(payload.get("epoch", -1)) != expected_epoch
            or float(payload["best_dev_metrics"]["average_precision"])
            != best_ap
        ):
            raise ValueError(
                f"{arm} sidecar checkpoint is not the earliest maximum-AP epoch."
            )
        loads.append(
            {
                **artifact(path, role=f"fresh_{arm}_sidecar_checkpoint"),
                "base_weights_sha256": weights_sha,
                "selected_on": "inner-train-only capability panel",
                "formal_inner_development_used": False,
                "run_config": artifact(
                    config_path, role=f"fresh_{arm}_sidecar_run_config"
                ),
                "metrics_history": artifact(
                    history_path, role=f"fresh_{arm}_sidecar_metrics_history"
                ),
                "external_load_allowed": False,
            }
        )
    for key in (
        "combined_initial_state_sha256",
        "sidecar_initial_state_sha256",
        "probe_initial_state_sha256",
    ):
        if sidecar_configs["p4"][key] != sidecar_configs["p5"][key]:
            raise ValueError(f"P4/P5 fresh initialization mismatch for {key}.")
    if (
        sidecar_configs["p4"]["dev_plan_sha256"]
        != sidecar_configs["p5"]["dev_plan_sha256"]
    ):
        raise ValueError("P4/P5 capability plan SHA mismatch.")
    if sidecar_train_plan_shas["p4"] != sidecar_train_plan_shas["p5"]:
        raise ValueError("P4/P5 train plan SHA sequence mismatch.")

    head_config_path = root / "event_balanced_heads_seed20260728/run_config.json"
    head_config = json.loads(head_config_path.read_text(encoding="utf-8"))
    if (
        head_config.get("clean_inner_replicate_exploratory") is not True
        or head_config.get("post_hoc_exploratory") is not False
        or int(head_config.get("epochs", -1)) != 3
    ):
        raise ValueError("Head run is not the formal fixed-three-epoch clean replicate.")
    if "all three matched arms run exactly epochs 1,2,3" not in str(
        head_config.get("epoch_protocol", "")
    ):
        raise ValueError("Head epoch protocol receipt is missing.")
    initial_by_arm = head_config["initial_state_sha256_by_arm"]
    if set(initial_by_arm) != {"p0", "p4", "p5"} or len(
        set(initial_by_arm.values())
    ) != 1:
        raise ValueError("P0/P4/P5 heads do not share one fresh initialization.")
    if next(iter(initial_by_arm.values())) != head_config["initial_state_sha256"]:
        raise ValueError("Head initial-state summary is inconsistent.")
    expected_head_caches = {
        arm: {
            "train_cache": root / f"sidecar/cache/{arm}/train.pt",
            "validation_cache": root / f"sidecar/cache/{arm}/val.pt",
        }
        for arm in ("p0", "p4", "p5")
    }
    for arm, path in heads.items():
        payload = cache.torch_load_trusted(path)
        if payload.get("arm") != arm:
            raise ValueError(f"{arm} head checkpoint arm mismatch.")
        cache_audit = payload.get("cache_audit", {})
        configured_cache_audit = head_config["cache_audits"][arm]
        if cache_audit != configured_cache_audit:
            raise ValueError(f"{arm} checkpoint/run-config cache audits differ.")
        history_path = path.parent / "metrics_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if [int(record["epoch"]) for record in history] != [1, 2, 3]:
            raise ValueError(f"{arm} head did not run exactly epochs 1,2,3.")
        best_ap = max(
            float(record["validation"]["event_balanced_ap"])
            for record in history
        )
        expected_best_epoch = next(
            int(record["epoch"])
            for record in history
            if float(record["validation"]["event_balanced_ap"]) == best_ap
        )
        if int(payload.get("epoch", -1)) != expected_best_epoch:
            raise ValueError(
                f"{arm} head checkpoint is not the earliest maximum-AP epoch."
            )
        for key, expected in expected_head_caches[arm].items():
            require_exact_audited_cache(
                cache_audit,
                key=key,
                expected=expected,
                formal_root=root,
                purpose=f"{arm} head {key}",
            )
        loads.append(
            {
                **artifact(path, role=f"fresh_{arm}_event_balanced_head"),
                "cache_audit": cache_audit,
                "selected_on": "formal inner development",
                "metrics_history": artifact(
                    history_path,
                    role=f"fresh_{arm}_head_metrics_history",
                ),
                "external_load_allowed": False,
            }
        )
    loads.append(
        {
            **artifact(
                head_config_path, role="fresh_p0_p4_p5_head_run_config"
            ),
            "shared_initial_state_sha256": head_config[
                "initial_state_sha256"
            ],
            "external_load_allowed": False,
        }
    )

    d1_config_path = root / "tempo_d1_three_seed/run_config.json"
    d1_config = json.loads(d1_config_path.read_text(encoding="utf-8"))
    configured_seeds = tuple(
        int(value.strip())
        for value in str(d1_config.get("seeds", "")).split(",")
        if value.strip()
    )
    if (
        d1_config.get("base_kind") != "event_balanced_p0"
        or configured_seeds != SEEDS
        or d1_config.get("resolved_arms") != ["p0_base", "d1_gated_delta"]
        or int(d1_config.get("epochs", -1)) != 4
        or int(d1_config.get("patience", -1)) != 1
    ):
        raise ValueError("D1 formal seed/arm/epoch protocol changed.")
    expected_p0 = heads["p0"].resolve()
    configured_p0 = safe_path(
        d1_config["event_base_checkpoint"], purpose="D1 P0 checkpoint"
    )
    if configured_p0 != expected_p0:
        raise ValueError("D1 did not load this lineage's P0 checkpoint.")
    for key, expected in (
        ("train_cache", root / "base_cache/train.pt"),
        ("dev_cache", root / "base_cache/val.pt"),
    ):
        configured = safe_path(d1_config[key], purpose=f"D1 {key}")
        if configured != expected.resolve():
            raise ValueError(f"D1 {key} is not the fresh P0 cache.")
    d1_cache_audit = d1_config["cache_audit"]
    require_exact_audited_cache(
        d1_cache_audit,
        key="train_cache",
        expected=base_caches["train"],
        formal_root=root,
        purpose="D1 audited train cache",
    )
    require_exact_audited_cache(
        d1_cache_audit,
        key="validation_cache",
        expected=base_caches["val"],
        formal_root=root,
        purpose="D1 audited validation cache",
    )
    expected_p0_sha = cache.sha256_file(expected_p0)
    expected_p0_state_sha = cache.state_dict_sha256(
        cache.torch_load_trusted(expected_p0)["model"]
    )
    for seed, path in d1_checkpoints.items():
        payload = cache.torch_load_trusted(path)
        if payload.get("arm") != "d1_gated_delta" or int(
            payload.get("seed", -1)
        ) != seed:
            raise ValueError(f"D1 seed {seed} checkpoint identity mismatch.")
        if safe_path(
            payload.get("frozen_base_checkpoint", ""),
            purpose=f"D1 seed {seed} frozen P0 parent",
        ) != expected_p0:
            raise ValueError(f"D1 seed {seed} frozen P0 parent path mismatch.")
        if payload.get("frozen_base_checkpoint_sha256") != expected_p0_sha:
            raise ValueError(f"D1 seed {seed} frozen P0 parent SHA mismatch.")
        if (
            payload.get("frozen_base_model_state_sha256")
            != expected_p0_state_sha
        ):
            raise ValueError(
                f"D1 seed {seed} frozen P0 model-state SHA mismatch."
            )
        seed_summary_path = path.parent / "summary.json"
        seed_summary = json.loads(seed_summary_path.read_text(encoding="utf-8"))
        frozen_audit = seed_summary["frozen_role_only_base_audit"]
        if (
            safe_path(
                frozen_audit["checkpoint"],
                purpose=f"D1 seed {seed} summary P0 parent",
            )
            != expected_p0
            or frozen_audit["checkpoint_sha256"] != expected_p0_sha
            or frozen_audit["model_state_sha256"] != expected_p0_state_sha
        ):
            raise ValueError(f"D1 seed {seed} summary parent closure mismatch.")
        loads.append(
            {
                **artifact(path, role=f"fresh_d1_seed_{seed}"),
                "loaded_base_checkpoint": str(expected_p0),
                "loaded_base_checkpoint_sha256": expected_p0_sha,
                "loaded_base_model_state_sha256": expected_p0_state_sha,
                "seed_summary": artifact(
                    seed_summary_path,
                    role=f"fresh_d1_seed_{seed}_summary",
                ),
                "external_load_allowed": False,
            }
        )

    payload = {
        "schema_version": "l89-clean-model-load-ledger-v1",
        "status": "finalized",
        "created_before_first_formal_model_load": True,
        "preload_plan_sha256": preload_sha,
        "preload_plan_path": str(preload_output),
        "formal_root": str(root),
        "allowed_external_model_loads": [
            {
                "path": str(weights),
                "sha256": weights_sha,
                "reason": "frozen Panopticon initialization explicitly reused",
            }
        ],
        "model_and_feature_loads": loads,
        "dependency_closure": {
            "frozen_panopticon_to_base_caches": [
                str(base_caches["train"]),
                str(base_caches["val"]),
            ],
            "fresh_base_train_to_sidecar_split": {
                "audit": str(split_audit_path),
                "audit_sha256": cache.sha256_file(split_audit_path),
                "fit_cache": str(root / "sidecar_split/train.pt"),
                "capability_cache": str(
                    root / "sidecar_split/capability.pt"
                ),
            },
            "sidecar_split_to_fresh_sidecars": {
                arm: str(path) for arm, path in sidecars.items()
            },
            "fresh_base_and_sidecars_to_representation_caches": {
                arm: [
                    str(root / f"sidecar/cache/{arm}/train.pt"),
                    str(root / f"sidecar/cache/{arm}/val.pt"),
                ]
                for arm in ("p0", "p4", "p5")
            },
            "fresh_representation_caches_to_heads": {
                arm: str(path) for arm, path in heads.items()
            },
            "fresh_base_cache_and_p0_head_to_d1": {
                "base_train": str(base_caches["train"]),
                "base_val": str(base_caches["val"]),
                "p0_head": str(heads["p0"]),
                "d1": {
                    str(seed): str(path)
                    for seed, path in d1_checkpoints.items()
                },
            },
        },
        "sidecar_train_only_lineage_verified": True,
        "sidecar_capability_eligible_negative_rows": int(
            selection["capability_eligible_negative_rows"]
        ),
        "sidecar_capability_plan_length": int(
            selection["locked_capability_plan_length"]
        ),
        "p4_p5_fresh_initialization_equal": True,
        "p4_p5_train_plan_sha_sequence_equal": True,
        "p4_p5_capability_plan_sha_equal": True,
        "p0_p4_p5_fresh_initialization_equal": True,
        "known_old_checkpoint_sha256_blocklist": sorted(
            KNOWN_OLD_CHECKPOINT_SHA256
        ),
        "known_old_checkpoint_sha_match": False,
        "old_sidecar_checkpoint_loaded": False,
        "old_head_checkpoint_loaded": False,
        "old_d1_checkpoint_loaded": False,
        "all_nonbase_model_loads_resolve_within_formal_root": True,
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache.atomic_json_write(final_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument(
        "--base-weights",
        default="/home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth",
    )
    parser.add_argument(
        "--phase", choices=("plan", "finalize"), default="finalize"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.phase == "plan":
        write_preload_plan(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
