#!/usr/bin/env python3
"""Create the original L89 temporal split with three difficult test events removed."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(
    "/mnt/engg-niulab/Yuyao/preprocessed_512/L89/"
    "l89_6time_temporal_16_resized_to_224"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "l89_6time_temporal_drop_3_hard_events"
DROP_EVENTS = {
    "tan20251229t042659c62s4001",
    "tan20251214t080904c51s4001",
    "tan20260119t183533c73s4001",
}


def event_key(plume_id: object) -> str:
    return re.sub(r"-[A-Za-z]+$", "", str(plume_id))


def filter_split(name: str) -> dict:
    source = SOURCE_DIR / f"L89_temporal_{name}.csv"
    data = pd.read_csv(source, low_memory=False)
    keys = data["plume_id"].map(event_key)
    removed = data.loc[keys.isin(DROP_EVENTS)].copy()
    kept = data.loc[~keys.isin(DROP_EVENTS)].copy()
    output = OUTPUT_DIR / f"L89_temporal_{name}_drop_3_hard_events.csv"
    removed_output = OUTPUT_DIR / f"L89_temporal_{name}_only_3_hard_events.csv"
    kept.to_csv(output, index=False)
    removed.to_csv(removed_output, index=False)
    return {
        "source": str(source),
        "output": str(output),
        "removed_output": str(removed_output),
        "rows_before": int(len(data)),
        "rows_after": int(len(kept)),
        "rows_removed": int(len(removed)),
        "removed_events": sorted(set(removed["plume_id"].map(event_key))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "method": "Preserve the original temporal train/test assignment, then remove the three events from both splits.",
        "drop_events": sorted(DROP_EVENTS),
        "train": filter_split("train"),
        "test": filter_split("test"),
    }
    (OUTPUT_DIR / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
