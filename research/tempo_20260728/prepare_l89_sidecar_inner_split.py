#!/usr/bin/env python3
"""Split formal inner-train into sidecar-fit and capability-panel artifacts.

Only the resolved formal inner-training CSV and its freshly extracted base CLS
cache are accepted. Whole canonical events are assigned to a deterministic
capability panel by taking the newest events until at least the requested row
count is reached. Those events are excluded from sidecar fitting.

The formal inner-development CSV/cache are not command-line inputs and are
never constructed, listed, statted, or read by this program.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import torch

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache
from research.pretraining_20260727 import rctp_l89_screen as screen


FORBIDDEN_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout|outer)([._/\\-]|$)", re.IGNORECASE
)
ROW_TENSOR_KEYS = (
    "features",
    "labels",
    "timestamps_utc_ns",
    "timestamp_valid_mask",
    "delta_days",
    "image_valid_mask",
    "valid_mask",
    "duplicate_mask",
    "duplicate_group_mask",
    "unique_mask",
    "valid_fraction",
    "load_status",
)
ROW_LIST_KEYS = ("ids", "plume_ids", "event_ids", "timestamps_utc_iso")


def safe_path(value: str | Path, *, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    if FORBIDDEN_RE.search(str(path)):
        raise ValueError(f"{purpose} contains a held-out marker: {path}")
    return path


def validate_source_alignment(
    frame: pd.DataFrame, payload: Mapping[str, Any], csv_path: Path
) -> None:
    cache.validate_cache_payload(
        payload, path=csv_path.with_suffix(".pt"), expected_split="train"
    )
    if cache.sha256_file(csv_path) != payload["csv_sha256"]:
        raise ValueError("Formal inner-train CSV SHA differs from base cache.")
    if len(frame) != len(payload["features"]):
        raise ValueError("Formal inner-train CSV/cache row counts differ.")
    ids = frame["id"].fillna("").astype(str).tolist()
    if ids != [str(value) for value in payload["ids"]]:
        raise ValueError("Formal inner-train CSV/cache ordered IDs differ.")
    plume_ids = frame["plume_id"].fillna("").astype(str).tolist()
    if plume_ids != [str(value) for value in payload["plume_ids"]]:
        raise ValueError("Formal inner-train CSV/cache ordered plume IDs differ.")
    declared_events = frame["event_group_id"].fillna("").astype(str).tolist()
    if declared_events != [str(value) for value in payload["event_ids"]]:
        raise ValueError("Formal inner-train CSV/cache ordered event IDs differ.")
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy()
    if not torch.equal(
        torch.as_tensor(labels, dtype=torch.long), payload["labels"].long()
    ):
        raise ValueError("Formal inner-train CSV/cache labels differ.")
    recomputed_label_sha = cache.tensor_sha256(payload["labels"])
    if recomputed_label_sha != payload.get("label_sha256"):
        raise ValueError("Formal inner-train cache label SHA is stale.")
    recomputed_timestamp_sha = cache.tensor_sha256(
        payload["timestamps_utc_ns"]
    )
    if recomputed_timestamp_sha != payload.get("timestamp_sha256"):
        raise ValueError("Formal inner-train cache timestamp SHA is stale.")


def row_tensor_digests(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the model-relevant row-tensor digests for an audit receipt."""

    keys = (
        "labels",
        "timestamps_utc_ns",
        "valid_mask",
        "unique_mask",
        "delta_days",
        "valid_fraction",
    )
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Cache is missing audit tensors {missing}.")
    return {
        f"{key}_sha256": cache.tensor_sha256(payload[key]) for key in keys
    }


def select_capability_events(
    frame: pd.DataFrame,
    eligible_negative: Sequence[bool],
    *,
    minimum_eligible_rows: int,
) -> tuple[list[str], pd.Timestamp]:
    if minimum_eligible_rows < 1:
        raise ValueError("minimum_eligible_rows must be positive.")
    if len(eligible_negative) != len(frame):
        raise ValueError("Eligibility mask does not align to the frame.")
    times = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    working = frame.assign(
        _event_time=times,
        _eligible_negative=pd.Series(
            eligible_negative, index=frame.index, dtype=bool
        ).astype(int),
    )
    summary = (
        working.groupby("event_group_id", observed=True)
        .agg(
            event_time=("_event_time", "max"),
            rows=("event_group_id", "size"),
            eligible_negative_rows=("_eligible_negative", "sum"),
        )
        .reset_index()
        .sort_values(
            ["event_time", "event_group_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    selected: list[str] = []
    eligible_rows = 0
    for record in summary.itertuples(index=False):
        selected.append(str(record.event_group_id))
        eligible_rows += int(record.eligible_negative_rows)
        if eligible_rows >= minimum_eligible_rows:
            break
    if eligible_rows < minimum_eligible_rows:
        raise ValueError(
            f"Only {eligible_rows} eligible negative rows are available for "
            f"a {minimum_eligible_rows}-row panel."
        )
    cutoff = summary.loc[
        summary["event_group_id"].astype(str).isin(selected), "event_time"
    ].min()
    return selected, cutoff


def subset_payload(
    source: Mapping[str, Any],
    indices: Sequence[int],
    *,
    split: str,
    csv_path: Path,
    frame: pd.DataFrame,
    source_cache_path: Path,
    panel_role: str,
    policy: str,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported cache split {split!r}.")
    selected = torch.as_tensor(list(indices), dtype=torch.long)
    result = copy.deepcopy(dict(source))
    source_rows = int(source["features"].shape[0])
    for key in ROW_TENSOR_KEYS:
        if key in source:
            value = source[key]
            if not isinstance(value, torch.Tensor) or value.shape[0] != source_rows:
                raise ValueError(f"Row tensor {key} is not aligned to source rows.")
            result[key] = value.index_select(0, selected)
    index_list = selected.tolist()
    for key in ROW_LIST_KEYS:
        if key in source:
            value = source[key]
            if len(value) != source_rows:
                raise ValueError(f"Row list {key} is not aligned to source rows.")
            result[key] = [value[index] for index in index_list]

    csv_sha = cache.sha256_file(csv_path)
    table_columns = [
        "id",
        "label",
        "plume_id",
        *result["path_columns"],
        *result["time_columns"],
    ]
    table_columns = [
        column for column in dict.fromkeys(table_columns) if column in frame
    ]
    table_sha = cache.input_table_sha256(frame, table_columns)
    contract = dict(result["input_contract"])
    contract.update(
        {
            "csv_sha256": csv_sha,
            "input_table_sha256": table_sha,
            "source_rows": source_rows,
            "selected_rows": int(len(frame)),
            "row_selection": policy,
            "row_selection_seed": 0,
        }
    )
    result.update(
        {
            "split": split,
            "input_contract": contract,
            "input_contract_sha256": cache.sha256_bytes(
                cache.canonical_json_bytes(contract)
            ),
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha,
            "input_table_sha256": table_sha,
            "feature_sha256": cache.tensor_sha256(result["features"]),
            "label_sha256": cache.tensor_sha256(result["labels"]),
            "timestamp_sha256": cache.tensor_sha256(
                result["timestamps_utc_ns"]
            ),
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "elapsed_seconds": 0.0,
            "subset_lineage": {
                "schema_version": "l89-sidecar-inner-train-subset-v1",
                "source_cache_path": str(source_cache_path),
                "source_cache_sha256": cache.sha256_file(source_cache_path),
                "source_csv_path": str(source["csv_path"]),
                "source_csv_sha256": str(source["csv_sha256"]),
                "panel_role": panel_role,
                "whole_canonical_events": True,
                "formal_inner_development_used": False,
            },
        }
    )
    return result


def write_cache(
    path: Path, payload: Mapping[str, Any], *, expected_split: str
) -> dict[str, Any]:
    cache.validate_cache_payload(
        payload, path=path, expected_split=expected_split
    )
    cache.atomic_torch_save(path, dict(payload))
    summary = {
        "path": str(path),
        "sha256": cache.sha256_file(path),
        "split": expected_split,
        "rows": int(payload["features"].shape[0]),
        "events": int(len(set(str(value) for value in payload["event_ids"]))),
        "feature_sha256": str(payload["feature_sha256"]),
        "row_tensor_digests": row_tensor_digests(payload),
        "input_contract_sha256": str(payload["input_contract_sha256"]),
        "source_cache_sha256": payload["subset_lineage"][
            "source_cache_sha256"
        ],
        "formal_inner_development_used": False,
    }
    cache.atomic_json_write(path.with_suffix(".pt.json"), summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_csv = safe_path(args.train_csv, purpose="formal inner-train CSV")
    train_cache = safe_path(args.train_cache, purpose="formal inner-train cache")
    output_root = safe_path(args.output_root, purpose="sidecar split output")
    if not train_csv.is_file():
        raise FileNotFoundError(train_csv)
    if not train_cache.is_file():
        raise FileNotFoundError(train_cache)
    if output_root.exists():
        raise FileExistsError(output_root)

    frame = pd.read_csv(train_csv, low_memory=False)
    required = {"id", "label", "event_group_id", "event_time"}
    required.add("plume_id")
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Formal inner-train CSV is missing {missing}.")
    source = cache.torch_load_trusted(train_cache)
    if not isinstance(source, Mapping):
        raise TypeError("Formal base cache must be a mapping.")
    validate_source_alignment(frame, source, train_csv)

    eligible_negative = (
        source["labels"].long().eq(0)
        & source["unique_mask"].bool().any(dim=1)
    ).cpu()
    capability_events, cutoff = select_capability_events(
        frame,
        eligible_negative.tolist(),
        minimum_eligible_rows=args.minimum_capability_rows,
    )
    capability_event_set = set(capability_events)
    capability_mask = frame["event_group_id"].astype(str).isin(
        capability_event_set
    )
    fit_frame = frame.loc[~capability_mask].copy().reset_index(drop=True)
    capability_frame = frame.loc[capability_mask].copy().reset_index(drop=True)
    fit_indices = frame.index[~capability_mask].tolist()
    capability_indices = frame.index[capability_mask].tolist()
    fit_eligible_negative = int(eligible_negative[~torch.as_tensor(
        capability_mask.to_numpy(), dtype=torch.bool
    )].sum())
    capability_eligible_negative = int(eligible_negative[torch.as_tensor(
        capability_mask.to_numpy(), dtype=torch.bool
    )].sum())
    if fit_eligible_negative < 2048:
        raise ValueError(
            f"Sidecar fit has only {fit_eligible_negative} eligible negative "
            "rows; 2048 are required."
        )
    if capability_eligible_negative < 512:
        raise ValueError(
            f"Capability panel has only {capability_eligible_negative} "
            "eligible negative rows; 512 are required."
        )
    overlap = set(fit_frame["event_group_id"].astype(str)) & set(
        capability_frame["event_group_id"].astype(str)
    )
    if overlap:
        raise RuntimeError("Sidecar fit/capability events overlap.")
    for name, subset in (
        ("fit", fit_frame),
        ("capability", capability_frame),
    ):
        if subset.empty:
            raise ValueError(f"{name} subset is empty.")
        if set(pd.to_numeric(subset["label"], errors="raise")) != {0, 1}:
            raise ValueError(f"{name} subset does not contain both labels.")

    output_root.mkdir(parents=True)
    fit_csv = output_root / "train.csv"
    capability_csv = output_root / "capability.csv"
    cache.atomic_csv_write(fit_csv, fit_frame)
    cache.atomic_csv_write(capability_csv, capability_frame)
    policy = (
        "whole-event newest-by-max-event-time capability panel until "
        f">={int(args.minimum_capability_rows)} eligible negative rows, where "
        "eligible=(label==0)&unique_mask.any(dim=1)"
    )
    fit_payload = subset_payload(
        source,
        fit_indices,
        split="train",
        csv_path=fit_csv,
        frame=fit_frame,
        source_cache_path=train_cache,
        panel_role="sidecar_fit",
        policy=policy + "; complement",
    )
    capability_payload = subset_payload(
        source,
        capability_indices,
        split="val",
        csv_path=capability_csv,
        frame=capability_frame,
        source_cache_path=train_cache,
        panel_role="sidecar_capability_selection",
        policy=policy,
    )
    renderer = screen.RendererConfig()
    train_plan_lengths = []
    for epoch in (0, 1):
        plan = screen.deterministic_reference_plan(
            fit_payload,
            seed=20260728,
            epoch=epoch,
            max_rows=2048,
            reference_label="negative",
            renderer=renderer,
        )
        train_plan_lengths.append(len(plan))
    capability_plan = screen.deterministic_reference_plan(
        capability_payload,
        seed=20260728 + 100_003,
        epoch=0,
        max_rows=512,
        reference_label="negative",
        renderer=renderer,
    )
    if train_plan_lengths != [2048, 2048]:
        raise RuntimeError(
            f"Sidecar train plan lengths are {train_plan_lengths}, not [2048, 2048]."
        )
    if len(capability_plan) != 512:
        raise RuntimeError(
            f"Sidecar capability plan length is {len(capability_plan)}, not 512."
        )
    fit_cache = output_root / "train.pt"
    capability_cache = output_root / "capability.pt"
    cache_summaries = {
        "train": write_cache(fit_cache, fit_payload, expected_split="train"),
        "capability": write_cache(
            capability_cache, capability_payload, expected_split="val"
        ),
    }
    # Exercise the exact downstream loader contract, including zero overlap
    # and comparable encoder/input contracts.
    _, _, pair_audit = cache.load_cache_pair(fit_cache, capability_cache)

    audit = {
        "schema_version": "l89-sidecar-inner-train-capability-v1",
        "source": {
            "train_csv": str(train_csv),
            "train_csv_sha256": cache.sha256_file(train_csv),
            "train_cache": str(train_cache),
            "train_cache_sha256": cache.sha256_file(train_cache),
            "rows": int(len(frame)),
            "events": int(frame["event_group_id"].astype(str).nunique()),
            "ordered_id_plume_event_label_exact": True,
            "recomputed_label_sha256": cache.tensor_sha256(
                source["labels"]
            ),
            "recomputed_timestamp_sha256": cache.tensor_sha256(
                source["timestamps_utc_ns"]
            ),
            "row_tensor_digests": row_tensor_digests(source),
        },
        "selection": {
            "policy": policy,
            "capability_minimum_eligible_negative_rows": int(
                args.minimum_capability_rows
            ),
            "capability_cutoff_utc": cutoff.isoformat(),
            "capability_events_in_selection_order": capability_events,
            "eligible_rule": "(label==0)&unique_mask.any(dim=1)",
            "sidecar_fit_eligible_negative_rows": fit_eligible_negative,
            "capability_eligible_negative_rows": (
                capability_eligible_negative
            ),
            "locked_train_plan_lengths_by_epoch": train_plan_lengths,
            "locked_capability_plan_length": len(capability_plan),
        },
        "sidecar_fit": {
            "rows": int(len(fit_frame)),
            "events": int(fit_frame["event_group_id"].nunique()),
            "csv": str(fit_csv),
            "csv_sha256": cache.sha256_file(fit_csv),
        },
        "capability_panel": {
            "rows": int(len(capability_frame)),
            "events": int(capability_frame["event_group_id"].nunique()),
            "csv": str(capability_csv),
            "csv_sha256": cache.sha256_file(capability_csv),
        },
        "event_overlap": 0,
        "cache_pair_audit": pair_audit,
        "cache_summaries": cache_summaries,
        "formal_inner_development_path_is_not_an_input": True,
        "formal_inner_development_used_for_sidecar_checkpoint_selection": False,
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    cache.atomic_json_write(output_root / "AUDIT.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minimum-capability-rows", type=int, default=512)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
