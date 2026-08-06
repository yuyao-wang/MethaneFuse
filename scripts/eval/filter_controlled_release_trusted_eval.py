#!/usr/bin/env python3
"""Remove explicitly untrusted events from a controlled-release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--exclude-value", action="append", required=True)
    parser.add_argument("--expected-before", type=int, required=True)
    parser.add_argument("--expected-after", type=int, required=True)
    parser.add_argument("--expected-positive-after", type=int, required=True)
    parser.add_argument("--expected-negative-after", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.input_manifest.resolve()
    frame = pd.read_csv(source, low_memory=False)
    if len(frame) != args.expected_before:
        raise ValueError(
            f"Expected {args.expected_before} source rows, found {len(frame)}"
        )
    if args.column not in frame:
        raise ValueError(f"Missing exclusion column: {args.column}")

    excluded_values = set(args.exclude_value)
    excluded = frame[frame[args.column].astype(str).isin(excluded_values)].copy()
    missing = sorted(excluded_values - set(excluded[args.column].astype(str)))
    if missing:
        raise ValueError(f"Requested exclusion values were absent: {missing}")
    kept = frame[~frame.index.isin(excluded.index)].copy()

    positives = int((kept["label"].astype(int) == 1).sum())
    negatives = int((kept["label"].astype(int) == 0).sum())
    if len(kept) != args.expected_after:
        raise ValueError(f"Expected {args.expected_after} kept rows, found {len(kept)}")
    if positives != args.expected_positive_after:
        raise ValueError(
            f"Expected {args.expected_positive_after} positives, found {positives}"
        )
    if negatives != args.expected_negative_after:
        raise ValueError(
            f"Expected {args.expected_negative_after} negatives, found {negatives}"
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.output_manifest, index=False)
    output = args.output_manifest.resolve()
    audit = {
        "input_manifest": str(source),
        "input_sha256": sha256(source),
        "output_manifest": str(output),
        "output_sha256": sha256(output),
        "exclusion_column": args.column,
        "exclusion_values": sorted(excluded_values),
        "excluded_rows": len(excluded),
        "excluded_label_counts": {
            str(key): int(value)
            for key, value in excluded["label"].astype(int).value_counts().sort_index().items()
        },
        "excluded_records": excluded[
            [column for column in ("id", "plume_id", "label", "datetime") if column in excluded]
        ].to_dict(orient="records"),
        "kept_rows": len(kept),
        "kept_positives": positives,
        "kept_negatives": negatives,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
