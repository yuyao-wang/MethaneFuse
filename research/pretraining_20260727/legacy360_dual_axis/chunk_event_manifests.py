#!/usr/bin/env python3
"""Split the two train shards into restart-safe event/plume components."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            keep, move = sorted((a, b))
            self.parent[move] = keep


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def component_positions(frame: pd.DataFrame) -> list[list[int]]:
    group_column = "event_id" if "event_id" in frame else "plume_id"
    union = UnionFind()
    for event, plume in zip(
        frame[group_column].astype(str), frame["plume_id"].astype(str)
    ):
        union.union(f"event:{event}", f"plume:{plume}")
    grouped: dict[str, list[int]] = {}
    for position, (event, plume) in enumerate(
        zip(frame[group_column].astype(str), frame["plume_id"].astype(str))
    ):
        root = union.find(f"event:{event}")
        if root != union.find(f"plume:{plume}"):
            raise AssertionError("event/plume union failed")
        grouped.setdefault(root, []).append(position)
    return sorted(grouped.values(), key=lambda values: values[0])


def pack_components(
    components: list[list[int]], target_rows: int
) -> list[list[int]]:
    chunks: list[list[int]] = []
    current: list[int] = []
    for component in components:
        if current and len(current) + len(component) > target_rows:
            chunks.append(sorted(current))
            current = []
        current.extend(component)
    if current:
        chunks.append(sorted(current))
    return chunks


def split_one(
    source: Path,
    output_dir: Path,
    *,
    gpu: int,
    target_rows: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    frame = pd.read_csv(
        source, dtype=str, keep_default_na=False, low_memory=False
    )
    required = {"id", "plume_id", "query360_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} misses {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{source}: duplicate id")
    queries = pd.to_numeric(frame["query360_index"], errors="raise").astype(
        np.int64
    )
    if queries.duplicated().any():
        raise ValueError(f"{source}: duplicate query360_index")

    packed = pack_components(component_positions(frame), target_rows)
    records: list[dict[str, object]] = []
    emitted_ids: set[str] = set()
    emitted_queries: set[int] = set()
    emitted_events: set[str] = set()
    emitted_plumes: set[str] = set()
    event_column = "event_id" if "event_id" in frame else "plume_id"
    for index, positions in enumerate(packed):
        chunk = frame.iloc[positions].copy()
        path = output_dir / (
            f"legacy360_train_core_gpu{gpu}_chunk{index:02d}.csv"
        )
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        chunk_ids = set(chunk["id"].astype(str))
        chunk_queries = set(
            pd.to_numeric(chunk["query360_index"]).astype(np.int64).tolist()
        )
        chunk_events = set(chunk[event_column].astype(str))
        chunk_plumes = set(chunk["plume_id"].astype(str))
        if (
            emitted_ids & chunk_ids
            or emitted_queries & chunk_queries
            or emitted_events & chunk_events
            or emitted_plumes & chunk_plumes
        ):
            raise AssertionError("a chunk split an id/query/event/plume group")
        atomic_csv(chunk, path)
        emitted_ids |= chunk_ids
        emitted_queries |= chunk_queries
        emitted_events |= chunk_events
        emitted_plumes |= chunk_plumes
        records.append(
            {
                "chunk": index,
                "path": str(path.absolute()),
                "sha256": sha256_file(path),
                "rows": int(len(chunk)),
                "events": int(len(chunk_events)),
                "plumes": int(len(chunk_plumes)),
                "query_min": int(min(chunk_queries)),
                "query_max": int(max(chunk_queries)),
            }
        )
    if emitted_ids != set(frame["id"].astype(str)):
        raise AssertionError("ID union differs from source")
    if emitted_queries != set(queries.tolist()):
        raise AssertionError("query360_index union differs from source")
    if sum(int(record["rows"]) for record in records) != len(frame):
        raise AssertionError("row union differs from source")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard0", required=True)
    parser.add_argument("--shard1", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-rows", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.target_rows < 1000:
        raise ValueError("--target-rows must be at least 1000")
    output = Path(args.output_dir).expanduser().absolute()
    sources = [
        Path(args.shard0).expanduser().absolute(),
        Path(args.shard1).expanduser().absolute(),
    ]
    result = {
        "schema_version": "legacy360-event-safe-chunks-v1",
        "target_rows": int(args.target_rows),
        "sources": [],
    }
    for gpu, source in enumerate(sources):
        records = split_one(
            source,
            output,
            gpu=gpu,
            target_rows=int(args.target_rows),
            overwrite=bool(args.overwrite),
        )
        result["sources"].append(
            {
                "gpu": gpu,
                "path": str(source),
                "sha256": sha256_file(source),
                "rows": int(sum(int(item["rows"]) for item in records)),
                "chunks": records,
            }
        )
    audit = output / "legacy360_train_chunk_audit.json"
    if audit.exists() and not args.overwrite:
        raise FileExistsError(audit)
    atomic_json(result, audit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
