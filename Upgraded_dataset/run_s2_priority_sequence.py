#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def wait_for_path(path: Path, timeout_seconds: int) -> None:
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for {path}")
        print(f"[Priority sequence] waiting for {path}", flush=True)
        time.sleep(30)


def stage_command(config: dict[str, object], stage: dict[str, object]) -> list[str]:
    common = config["common"]
    command = [
        str(common["python"]),
        str(common["queue_script"]),
        "--side",
        str(stage["side"]),
        "--split-root",
        str(stage["split_root"]),
        "--checkpoint-root",
        str(stage["checkpoint_root"]),
        "--result-root",
        str(stage["result_root"]),
        "--run-prefix",
        str(stage["run_prefix"]),
        "--python",
        str(common["python"]),
        "--train-script",
        str(common["train_script"]),
        "--weights",
        str(common["weights"]),
        "--max-epochs",
        str(common.get("max_epochs", 10)),
        "--min-epoch",
        str(common.get("min_epoch", 8)),
        "--patience",
        str(common.get("patience", 2)),
        "--min-delta",
        str(common.get("min_delta", 0.001)),
        "--early-stop-metric",
        str(common.get("early_stop_metric", "test_auroc")),
        "--batch-size",
        str(common.get("batch_size", 32)),
        "--num-workers",
        str(common.get("num_workers", 4)),
        "--splits",
        *[str(value) for value in stage["splits"]],
    ]
    if int(stage.get("seed", -1)) >= 0:
        command.extend(["--seed", str(stage["seed"])])
    if int(stage.get("contract_downsample_size", 0)) > 0:
        command.extend(
            [
                "--contract-downsample-size",
                str(stage["contract_downsample_size"]),
            ]
        )
    if stage.get("zero_t0_band_indices"):
        command.extend(
            [
                "--zero-t0-band-indices",
                str(stage["zero_t0_band_indices"]),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    status_path = Path(args.status)
    status: dict[str, object] = {
        "state": "running",
        "config": args.config,
        "started_at": utc_now(),
        "completed_stages": [],
    }
    atomic_json(status_path, status)
    try:
        for index, stage in enumerate(config["stages"], start=1):
            wait_for = stage.get("wait_for")
            if wait_for:
                wait_for_path(
                    Path(str(wait_for)),
                    int(stage.get("wait_timeout_seconds", 21600)),
                )
            command = stage_command(config, stage)
            status.update(
                {
                    "current_stage": stage["name"],
                    "current_stage_index": index,
                    "current_command": command,
                    "updated_at": utc_now(),
                }
            )
            atomic_json(status_path, status)
            print(
                f"[Priority sequence] stage {index}/{len(config['stages'])}: "
                f"{stage['name']}",
                flush=True,
            )
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(
                    f"stage {stage['name']} failed with {completed.returncode}"
                )
            status["completed_stages"].append(
                {
                    "name": stage["name"],
                    "finished_at": utc_now(),
                }
            )
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)
        status.update(
            {
                "state": "complete",
                "current_stage": None,
                "finished_at": utc_now(),
            }
        )
        atomic_json(status_path, status)
    except Exception as exc:
        status.update(
            {
                "state": "failed",
                "error": repr(exc),
                "finished_at": utc_now(),
            }
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
