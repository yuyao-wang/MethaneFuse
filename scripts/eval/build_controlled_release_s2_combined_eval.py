#!/usr/bin/env python3
"""Combine audited controlled-release S2 positive and negative manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path, expected_label: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    for row in rows:
        if int(row["label"]) != expected_label:
            raise ValueError(
                f"Expected label {expected_label} in {path}, got {row['label']}"
            )
        for column in ("s2_0_path", "s2_90_path", "s2_360_path"):
            value = row.get(column, "")
            if not value or not Path(value).is_file():
                raise FileNotFoundError(f"Missing {column} for {row.get('id')}: {value}")
    return fieldnames, rows


def main() -> None:
    args = parse_args()
    pos_fields, positives = read_rows(args.positive_manifest, 1)
    neg_fields, negatives = read_rows(args.negative_manifest, 0)
    if pos_fields != neg_fields:
        raise ValueError("Positive and negative manifest headers differ")

    fieldnames = [*pos_fields, "source_group"]
    combined: list[dict[str, str]] = []
    for source_group, rows in (
        ("original_table_label1", positives),
        ("original_table_label0", negatives),
    ):
        for row in rows:
            combined.append({**row, "source_group": source_group})

    ids = [row["id"] for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Combined manifest contains duplicate IDs")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(
        {
            "output": str(args.output_manifest),
            "rows": len(combined),
            "label1": len(positives),
            "label0": len(negatives),
        }
    )


if __name__ == "__main__":
    main()
