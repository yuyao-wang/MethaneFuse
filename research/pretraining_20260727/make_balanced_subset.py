#!/usr/bin/env python3
"""Create deterministic class-balanced CSV subsets for short pilot runs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=/absolute/path.csv")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_absolute():
        raise argparse.ArgumentTypeError("Expected nonempty label and absolute path")
    return label, path


def stable_rank(row: pd.Series, columns: list[str], seed: int) -> str:
    identity = "\x1f".join(str(row[column]) for column in columns)
    payload = f"{seed}\x1e{identity}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--per_class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--key_columns",
        default="",
        help="Comma-separated stable identity columns; auto-detected when empty.",
    )
    args = parser.parse_args()
    if args.per_class < 1:
        raise ValueError("--per_class must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, input_path in args.input:
        frame = pd.read_csv(input_path, low_memory=False)
        if args.label_column not in frame.columns:
            raise ValueError(f"{input_path} lacks {args.label_column}")
        requested_keys = [
            value.strip() for value in args.key_columns.split(",") if value.strip()
        ]
        if requested_keys:
            missing = [column for column in requested_keys if column not in frame.columns]
            if missing:
                raise ValueError(f"{input_path} lacks key columns {missing}")
            key_columns = requested_keys
        else:
            candidates = (
                "sample_id",
                "id",
                "image_path",
                "data_path",
                "path",
                "plume_id",
            )
            key_columns = [column for column in candidates if column in frame.columns]
            if not key_columns:
                key_columns = [frame.columns[0]]

        selected = []
        counts = {}
        for class_value, class_frame in frame.groupby(args.label_column, sort=True):
            ranked = class_frame.copy()
            ranked["_pilot_rank"] = ranked.apply(
                stable_rank, axis=1, columns=key_columns, seed=args.seed
            )
            take = min(args.per_class, len(ranked))
            ranked = ranked.sort_values("_pilot_rank").head(take).drop(columns="_pilot_rank")
            selected.append(ranked)
            counts[str(class_value)] = int(take)
        output = pd.concat(selected, ignore_index=True)
        output["_pilot_shuffle"] = output.apply(
            stable_rank, axis=1, columns=key_columns, seed=args.seed + 1
        )
        output = (
            output.sort_values("_pilot_shuffle")
            .drop(columns="_pilot_shuffle")
            .reset_index(drop=True)
        )
        output_path = args.output_dir / f"{label}.csv"
        output.to_csv(output_path, index=False)
        print(
            f"[balanced-subset] {label}: input={len(frame)} output={len(output)} "
            f"class_counts={counts} keys={key_columns} -> {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
