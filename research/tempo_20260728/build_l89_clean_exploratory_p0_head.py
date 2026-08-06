#!/usr/bin/env python3
"""Build one formal-equivalent clean-inner P0 head for a bounded D1 screen.

This is an isolated exploratory sidecar.  It imports the frozen matched-head
implementation but trains only its P0 arm, avoiding the two irrelevant
duplicate arms that the three-arm formal comparison requires.  The model
initialization, event/class-balanced loss, batches, optimizer, three epochs,
checkpoint selection, and threshold selection are exactly the P0 settings in
``run_l89_clean_replicate.sh``.

Only train/development cache paths are accepted.  No test, sealed, holdout, or
outer path is readable by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import torch

from research.pretraining_20260727 import (
    l89_ragged_cls_experiment as cache_runner,
)
from research.pretraining_20260727 import (
    rctp_l89_event_balanced_head_followup as head_runner,
)


SCRIPT_VERSION = "l89-clean-exploratory-p0-only-v1"
FORBIDDEN_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout|outer)([._/\\-]|$)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_forbidden(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    if FORBIDDEN_RE.search(str(resolved)):
        raise ValueError(f"{purpose} is held-out-like and forbidden: {resolved}")
    cache_runner.assert_not_sealed_path(resolved, purpose=purpose)
    return resolved


def take(cache: Mapping[str, Any]) -> dict[str, Any]:
    return cache_runner.take_rows(cache, cache_runner.select_usable_rows(cache))


def run(args: argparse.Namespace) -> None:
    train_path = refuse_forbidden(Path(args.train_cache), purpose="P0 train cache")
    dev_path = refuse_forbidden(Path(args.dev_cache), purpose="P0 dev cache")
    output_dir = refuse_forbidden(Path(args.output_dir), purpose="P0 output")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P0 output: {output_dir}")
    if args.epochs != 3:
        raise ValueError("Formal-equivalent P0 is frozen to exactly three epochs.")
    if args.seed != 20260728:
        raise ValueError("Formal-equivalent P0 seed is frozen to 20260728.")

    torch.set_num_threads(int(args.num_threads))
    train_cache, dev_cache, cache_audit = cache_runner.load_cache_pair(
        train_path, dev_path
    )
    if int(cache_audit["event_overlap"]) != 0:
        raise ValueError("P0 train/development canonical events overlap.")
    if int(cache_audit["feature_dim"]) != 1536:
        raise ValueError("Exact P0 cache must have [base768, zero768] width.")
    train = take(train_cache)
    dev = take(dev_cache)
    feature_dim = int(train["features"].shape[-1])
    num_roles = int(train["features"].shape[1])
    t0_index = int(train_cache["t0_index"])
    role_index = train_cache["role_index"].long()
    if num_roles != 6 or t0_index != 0:
        raise ValueError("Clean L89 P0 requires six roles with t0 at index zero.")
    if not torch.equal(role_index, dev_cache["role_index"].long()):
        raise ValueError("P0 train/development role indices differ.")
    residual_nonzero_train = int(
        torch.count_nonzero(train["features"][..., 768:]).item()
    )
    residual_nonzero_dev = int(
        torch.count_nonzero(dev["features"][..., 768:]).item()
    )
    if residual_nonzero_train or residual_nonzero_dev:
        raise ValueError("P0 residual channel is not exactly zero.")

    periods_days = tuple(
        float(value.strip())
        for value in args.delta_periods.split(",")
        if value.strip()
    )
    initial_state, signature = head_runner.build_initial_state(
        feature_dim=feature_dim,
        num_roles=num_roles,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        periods_days=periods_days,
        t0_index=t0_index,
        seed=args.seed,
    )
    initial_sha = cache_runner.state_dict_sha256(initial_state)
    batch_plan = {
        str(epoch): [
            indices.tolist()
            for indices in cache_runner.fixed_epoch_batches(
                len(train["labels"]),
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=epoch,
                shuffle=True,
            )
        ]
        for epoch in range(1, args.epochs + 1)
    }
    batch_plan_sha = cache_runner.sha256_bytes(
        cache_runner.canonical_json_bytes(batch_plan)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "script_version": SCRIPT_VERSION,
        "scope": "bounded clean-inner exploratory P0 parent for D1 screen",
        "formal_equivalence": {
            "imported_training_function": (
                "rctp_l89_event_balanced_head_followup.train_arm"
            ),
            "arm": "p0",
            "same_initialization_seed": True,
            "same_optimizer_and_batches": True,
            "same_three_epoch_cap": True,
            "same_event_balanced_ap_selection": True,
            "same_event_balanced_macro_f1_threshold": True,
        },
        "train_cache": str(train_path),
        "dev_cache": str(dev_path),
        "train_cache_sha256": sha256_file(train_path),
        "dev_cache_sha256": sha256_file(dev_path),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "model_dim": int(args.model_dim),
        "num_heads": int(args.num_heads),
        "mlp_ratio": float(args.mlp_ratio),
        "dropout": float(args.dropout),
        "delta_periods": args.delta_periods,
        "periods_days": list(periods_days),
        "num_threads": int(args.num_threads),
        "initial_state_sha256": initial_sha,
        "batch_plan_sha256": batch_plan_sha,
        "cache_audit": cache_audit,
        "p0_zero_residual_nonzero_count": {
            "train": residual_nonzero_train,
            "dev": residual_nonzero_dev,
        },
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache_runner.atomic_json_write(output_dir / "run_config.json", run_config)
    cache_runner.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "script_version": SCRIPT_VERSION,
            "test_or_sealed_or_holdout_or_outer_read": False,
        },
    )
    train_args = SimpleNamespace(
        clean_inner_replicate=True,
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        periods_days=periods_days,
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
    )
    try:
        history, best, predictions = head_runner.train_arm(
            "p0",
            train_data=train,
            val_data=dev,
            role_index=role_index,
            t0_index=t0_index,
            initial_state=initial_state,
            parameter_signature=signature,
            args=train_args,
            arm_dir=output_dir / "p0",
            cache_audit=cache_audit,
        )
        checkpoint_path = (
            output_dir / "p0" / "checkpoint_best_event_balanced_ap.pt"
        )
        prediction_path = (
            output_dir
            / "p0"
            / "validation_best_event_balanced_ap_predictions.csv"
        )
        result = {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "best": best,
            "epochs_observed": len(history),
            "prediction_rows": int(len(predictions)),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_csv": str(prediction_path),
            "prediction_csv_sha256": sha256_file(prediction_path),
            "initial_state_sha256": initial_sha,
            "batch_plan_sha256": batch_plan_sha,
            "test_or_sealed_or_holdout_or_outer_read": False,
        }
        cache_runner.atomic_json_write(output_dir / "RESULT.json", result)
        cache_runner.atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "script_version": SCRIPT_VERSION,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "test_or_sealed_or_holdout_or_outer_read": False,
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        cache_runner.atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "script_version": SCRIPT_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "test_or_sealed_or_holdout_or_outer_read": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--num-threads", type=int, default=12)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
