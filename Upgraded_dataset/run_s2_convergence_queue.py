#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("row_random_80_20", "event_disjoint_80_20")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_metrics(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    epochs = [int(record["epoch"]) for record in records]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"metrics epochs are not unique and sorted: {path}: {epochs}")
    return records


def early_stop_state(
    records: list[dict[str, object]],
    metric: str,
    min_epoch: int,
    min_delta: float,
) -> tuple[float, int]:
    best = float("-inf")
    bad_epochs = 0
    for record in records:
        value = float(record[metric])
        epoch = int(record["epoch"])
        if value > best + min_delta:
            best = value
            bad_epochs = 0
        elif epoch >= min_epoch:
            bad_epochs += 1
    return best, bad_epochs


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def stream_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n[Queue command] {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def run_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    train_csv = Path(args.split_root) / split / "train.csv"
    test_csv = Path(args.split_root) / split / "test.csv"
    metrics_path = Path(args.result_root) / f"{split}.metrics.jsonl"
    log_path = Path(args.result_root) / f"{split}.convergence.log"
    status_path = Path(args.result_root) / f"{split}.status.json"
    run_name = f"{args.run_prefix}_{split}_full_finetune"
    latest_path = Path(args.checkpoint_root) / run_name / "ckpt_latest.pth"
    records = read_metrics(metrics_path)
    current_epoch = int(records[-1]["epoch"]) if records else 0
    if current_epoch and not latest_path.is_file():
        raise FileNotFoundError(
            f"metrics resume at epoch {current_epoch}, but checkpoint is missing: {latest_path}"
        )
    if not current_epoch and latest_path.exists():
        raise RuntimeError(f"checkpoint exists without metrics: {latest_path}")
    # A finished split is either at max_epochs or was early stopped. Honouring the
    # recorded early_stopped state matters because otherwise re-running the queue
    # (after a crash or a config swap) silently trains past the early-stop
    # decision, which would turn a stopped run into a longer, differently
    # selected one.
    previously_early_stopped = False
    if status_path.is_file():
        with suppress(Exception):
            previously_early_stopped = (
                json.loads(status_path.read_text()).get("state") == "early_stopped"
            )
    if current_epoch >= args.max_epochs or previously_early_stopped:
        return {
            "split": split,
            "state": "already_complete",
            "reason": "early_stopped" if previously_early_stopped else "max_epochs",
            "last_epoch": current_epoch,
            "metrics_path": str(metrics_path),
            "checkpoint": str(latest_path),
        }

    best, bad_epochs = early_stop_state(
        records, args.early_stop_metric, args.min_epoch, args.min_delta
    )
    status: dict[str, object] = {
        "split": split,
        "state": "running",
        "started_at": utc_now(),
        "side": args.side,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "checkpoint_root": args.checkpoint_root,
        "run_name": run_name,
        "metrics_path": str(metrics_path),
        "early_stop": {
            "metric": args.early_stop_metric,
            "mode": "maximize",
            "min_epoch": args.min_epoch,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "resume_epoch": current_epoch,
        "best_metric": best if best != float("-inf") else None,
        "bad_epochs": bad_epochs,
    }
    write_status(status_path, status)

    env = os.environ.copy()
    env.setdefault("XFORMERS_DISABLED", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    for target_epoch in range(current_epoch + 1, args.max_epochs + 1):
        command = [
            args.python,
            args.train_script,
            "--train_csv",
            str(train_csv),
            "--test_csv",
            str(test_csv),
            "--weights",
            args.weights,
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(target_epoch),
            "--head_lr",
            "0.001",
            "--backbone_lr",
            "0.0001",
            "--lr_scheduler",
            "noam",
            "--warmup_steps",
            "4000",
            "--weight_decay",
            "0.0005",
            "--momentum",
            "0.9",
            "--num_workers",
            str(args.num_workers),
            "--pad_to_multiple",
            "1",
            "--input_resize_size",
            "224",
            "--temporal_mode",
            "three",
            "--device",
            "cuda:0",
            "--log_interval",
            "100",
            "--t0_col",
            "s2_0_path",
            "--t90_col",
            "s2_90_path",
            "--t360_col",
            "s2_360_path",
            "--train_backbone",
            "--freeze_backbone_epochs",
            "0",
            "--max_grad_norm",
            "1.0",
            "--save_checkpoints",
            "--checkpoint_dir",
            args.checkpoint_root,
            "--run_name",
            run_name,
            "--metrics_jsonl",
            str(metrics_path),
        ]
        if args.contract_downsample_size > 0:
            command.extend(
                [
                    "--contract_downsample_size",
                    str(args.contract_downsample_size),
                ]
            )
        if args.zero_t0_band_indices:
            command.extend(
                [
                    "--zero_t0_band_indices",
                    str(args.zero_t0_band_indices),
                ]
            )
        if args.seed >= 0:
            command.extend(["--seed", str(args.seed)])
        if target_epoch > 1:
            command.append("--resume")
        status.update(
            {
                "state": "running",
                "target_epoch": target_epoch,
                "updated_at": utc_now(),
            }
        )
        write_status(status_path, status)
        exit_code = stream_command(command, log_path, env)
        if exit_code:
            status.update(
                {
                    "state": "failed",
                    "exit_code": exit_code,
                    "failed_epoch": target_epoch,
                    "finished_at": utc_now(),
                }
            )
            write_status(status_path, status)
            raise RuntimeError(f"{split} epoch {target_epoch} failed with {exit_code}")
        records = read_metrics(metrics_path)
        if not records or int(records[-1]["epoch"]) != target_epoch:
            raise RuntimeError(f"missing epoch {target_epoch} metric in {metrics_path}")
        best, bad_epochs = early_stop_state(
            records, args.early_stop_metric, args.min_epoch, args.min_delta
        )
        status.update(
            {
                "completed_epoch": target_epoch,
                "best_metric": best,
                "bad_epochs": bad_epochs,
                "last_metrics": records[-1],
                "updated_at": utc_now(),
            }
        )
        write_status(status_path, status)
        if target_epoch >= args.min_epoch and bad_epochs >= args.patience:
            status.update(
                {
                    "state": "early_stopped",
                    "finished_at": utc_now(),
                    "last_epoch": target_epoch,
                }
            )
            write_status(status_path, status)
            return status

    status.update(
        {
            "state": "max_epochs_reached",
            "finished_at": utc_now(),
            "last_epoch": args.max_epochs,
        }
    )
    write_status(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("legacy", "gee"), required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--train-script", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--min-epoch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-metric", default="test_auroc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Split execution order; defaults to row-random followed by event-disjoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Non-negative paired-run seed forwarded to the trainer.",
    )
    parser.add_argument(
        "--contract-downsample-size",
        type=int,
        default=0,
        help="Explicit spatial bottleneck forwarded to the trainer.",
    )
    parser.add_argument(
        "--zero-t0-band-indices",
        default="",
        help="Comma-separated t0 band indices to zero, forwarded to the trainer.",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    queue_status = {
        "state": "running",
        "side": args.side,
        "started_at": utc_now(),
        "splits": [],
    }
    queue_path = result_root / "queue_status.json"
    write_status(queue_path, queue_status)
    try:
        for split in args.splits:
            result = run_split(args, split)
            queue_status["splits"].append(result)
            queue_status["updated_at"] = utc_now()
            write_status(queue_path, queue_status)
    except Exception as exc:
        queue_status.update(
            {
                "state": "failed",
                "error": repr(exc),
                "finished_at": utc_now(),
            }
        )
        write_status(queue_path, queue_status)
        raise
    queue_status.update({"state": "complete", "finished_at": utc_now()})
    write_status(queue_path, queue_status)
    print(json.dumps(queue_status, indent=2))


if __name__ == "__main__":
    main()
