#!/usr/bin/env python3
"""Combine trusted positives with newly materialized temporal negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-source-manifest", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def require_paths(frame: pd.DataFrame, name: str) -> None:
    for row in frame.itertuples(index=False):
        for column in ("s2_0_path", "s2_90_path", "s2_360_path"):
            path = Path(str(getattr(row, column)))
            if not path.is_file():
                raise FileNotFoundError(f"{name}/{row.id}/{column}: {path}")


def main() -> None:
    args = parse_args()
    positives = pd.read_csv(args.positive_source_manifest, low_memory=False)
    positives = positives[pd.to_numeric(positives["label"], errors="raise").eq(1)].copy()
    negatives = pd.read_csv(args.negative_manifest, low_memory=False)
    if not pd.to_numeric(negatives["label"], errors="raise").eq(0).all():
        raise ValueError("negative manifest contains a nonzero label")
    require_paths(positives, "positive")
    require_paths(negatives, "negative")
    positives["source_group"] = "trusted_controlled_release_positive"
    negatives["source_group"] = "random_time_no_known_controlled_release"
    columns = list(dict.fromkeys([*positives.columns, *negatives.columns]))
    output = pd.concat(
        [positives.reindex(columns=columns), negatives.reindex(columns=columns)],
        ignore_index=True,
    )
    if output["id"].astype(str).duplicated().any():
        raise ValueError("duplicate IDs in combined manifest")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_manifest, index=False)
    print(
        {
            "output": str(args.output_manifest),
            "rows": len(output),
            "positives": len(positives),
            "negatives": len(negatives),
        }
    )


if __name__ == "__main__":
    main()
