#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    path = Path(args.csv)
    frame = pd.read_csv(path, low_memory=False)
    reset = 0
    for timepoint in TIMEPOINTS:
        status_column = f"{timepoint}_recrop_status"
        message_column = f"{timepoint}_recrop_message"
        frame[status_column] = frame[status_column].astype(object)
        frame[message_column] = frame[message_column].astype(object)
        failed = (
            frame[status_column]
            .fillna("")
            .astype(str)
            .eq("failed")
        )
        reset += int(failed.sum())
        frame.loc[failed, status_column] = ""
        frame.loc[failed, message_column] = ""
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    print(f"rows={len(frame)} reset_failed={reset}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
