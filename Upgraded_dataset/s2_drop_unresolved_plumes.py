#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def save_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-report", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--resume-csv", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repair = json.loads(Path(args.repair_report).read_text())
    unresolved = repair.get("unresolved", [])
    plume_ids = sorted(
        {
            str(record["plume_id"])
            for record in unresolved
        }
    )
    if not plume_ids:
        raise RuntimeError("repair report has no unresolved plumes")

    source_path = Path(args.source_csv)
    resume_path = Path(args.resume_csv)
    source = pd.read_csv(source_path, low_memory=False)
    resume = pd.read_csv(resume_path, low_memory=False)
    if len(source) != len(resume):
        raise RuntimeError(
            f"row mismatch source={len(source)} resume={len(resume)}"
        )

    source_drop = source["plume_id"].astype(str).isin(plume_ids)
    resume_drop = resume["plume_id"].astype(str).isin(plume_ids)
    if int(source_drop.sum()) != len(plume_ids):
        raise RuntimeError(
            "not every unresolved plume occurs exactly once in source"
        )
    if int(resume_drop.sum()) != len(plume_ids):
        raise RuntimeError(
            "not every unresolved plume occurs exactly once in resume"
        )

    dropped_rows = source.loc[
        source_drop,
        [
            "plume_id",
            "plume_latitude",
            "plume_longitude",
        ],
    ].to_dict("records")
    source = source.loc[~source_drop].reset_index(drop=True)
    resume = resume.loc[~resume_drop].reset_index(drop=True)
    save_atomic(source, source_path)
    save_atomic(resume, resume_path)

    report = {
        "reason": (
            "No same-acquisition CDSE product set could provide a "
            "fully covered 512x512 point-centered crop."
        ),
        "dropped_plumes": len(plume_ids),
        "remaining_rows": len(source),
        "plume_ids": plume_ids,
        "rows": dropped_rows,
        "unresolved_timepoints": unresolved,
    }
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
