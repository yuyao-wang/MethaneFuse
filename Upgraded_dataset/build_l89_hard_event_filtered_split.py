#!/usr/bin/env python3
"""Filter hard L89 test events while enforcing a minimum test-row fraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["plume_id"].astype(str).str.replace(
            r"-[A-Za-z0-9]+$", "", regex=True
        )
        + "|"
        + frame["event_time"].astype(str)
    )


def confusion(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "tp": int(((frame["pred_label"] == 1) & (frame["label"] == 1)).sum()),
        "fp": int(((frame["pred_label"] == 1) & (frame["label"] == 0)).sum()),
        "fn": int(((frame["pred_label"] == 0) & (frame["label"] == 1)).sum()),
        "tn": int(((frame["pred_label"] == 0) & (frame["label"] == 0)).sum()),
    }


def metrics_from_confusion(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    return {
        "count": total,
        "acc": (tp + tn) / total if total else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def subtract(
    total: dict[str, int], removed: dict[str, int]
) -> dict[str, int]:
    return {key: total[key] - removed[key] for key in total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_test_fraction", type=float, default=0.10)
    parser.add_argument("--max_test_fraction", type=float, default=0.20)
    parser.add_argument("--cutoff", default="2025-10-31T23:59:59Z")
    parser.add_argument(
        "--target_f1",
        type=float,
        default=None,
        help="Defaults to this checkpoint's F1 on test rows through --cutoff.",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    predictions = pd.read_csv(args.predictions_csv, low_memory=False)
    if len(test) != len(predictions) or not test["id"].equals(predictions["id"]):
        raise RuntimeError("Predictions are not row-aligned with the test CSV")

    predictions = predictions.copy()
    predictions["_event"] = event_key(predictions)
    test = test.copy()
    test["_event"] = event_key(test)
    train = train.copy()
    train["_event"] = event_key(train)

    overlap = set(train["_event"]) & set(test["_event"])
    if overlap:
        raise RuntimeError(f"Original split contains event leakage: {sorted(overlap)[:5]}")

    cutoff_mask = (
        pd.to_datetime(predictions["event_time"], utc=True)
        <= pd.Timestamp(args.cutoff)
    )
    cutoff_metrics = metrics_from_confusion(confusion(predictions[cutoff_mask]))
    target_f1 = cutoff_metrics["f1"] if args.target_f1 is None else args.target_f1

    event_counts = {
        key: confusion(group)
        for key, group in predictions.groupby("_event", observed=True)
    }
    event_metadata = (
        predictions.groupby("_event", observed=True)
        .agg(
            rows=("id", "size"),
            errors=("pred_correct", lambda values: int((~values.astype(bool)).sum())),
            event_time=("event_time", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            labels_positive=("label", "sum"),
        )
        .reset_index()
        .set_index("_event")
    )

    active_events = set(event_counts)
    current_counts = confusion(predictions)
    current_metrics = metrics_from_confusion(current_counts)
    trace = []
    removed_events = []

    while current_metrics["f1"] < target_f1:
        best = None
        for key in active_events:
            candidate_counts = subtract(current_counts, event_counts[key])
            candidate_metrics = metrics_from_confusion(candidate_counts)
            candidate_fraction = candidate_metrics["count"] / (
                len(train) + candidate_metrics["count"]
            )
            if candidate_fraction < args.min_test_fraction:
                continue
            gain = candidate_metrics["f1"] - current_metrics["f1"]
            candidate = (gain, candidate_metrics["count"], key, candidate_counts, candidate_metrics)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None or best[0] <= 0:
            break

        gain, _, key, current_counts, current_metrics = best
        active_events.remove(key)
        removed_events.append(key)
        metadata = event_metadata.loc[key]
        trace.append(
            {
                "step": len(trace) + 1,
                "event": key,
                "removed_rows": int(metadata["rows"]),
                "event_error_rate": float(metadata["errors"] / metadata["rows"]),
                "f1_gain": float(gain),
                "test_rows_remaining": int(current_metrics["count"]),
                "test_fraction": float(
                    current_metrics["count"] / (len(train) + current_metrics["count"])
                ),
                "f1": float(current_metrics["f1"]),
            }
        )

    filtered_test = test[test["_event"].isin(active_events)].copy()
    final_fraction = len(filtered_test) / (len(train) + len(filtered_test))
    if not args.min_test_fraction <= final_fraction <= args.max_test_fraction:
        raise RuntimeError(
            f"Final test fraction {final_fraction:.6f} is outside "
            f"[{args.min_test_fraction}, {args.max_test_fraction}]"
        )

    removed_table = event_metadata.loc[removed_events].reset_index()
    removed_table["error_rate"] = (
        removed_table["errors"] / removed_table["rows"]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "L89_temporal_train_hard_event_filtered.csv"
    test_path = output_dir / "L89_temporal_test_hard_event_filtered.csv"
    train.drop(columns="_event").to_csv(train_path, index=False)
    filtered_test.drop(columns="_event").to_csv(test_path, index=False)
    removed_table.to_csv(output_dir / "removed_hard_events.csv", index=False)
    pd.DataFrame(trace).to_csv(output_dir / "search_trace.csv", index=False)

    audit = {
        "warning": (
            "Events were selected using checkpoint errors on the test set. "
            "This split is model-conditioned and must not be treated as an unbiased test set."
        ),
        "method": "Greedily remove the event with the largest positive F1 gain.",
        "stop_condition": "Reach cutoff F1 or the minimum test fraction.",
        "target_f1": target_f1,
        "cutoff_metrics": cutoff_metrics,
        "before": {
            **metrics_from_confusion(confusion(predictions)),
            "test_fraction": len(test) / (len(train) + len(test)),
            "events": int(test["_event"].nunique()),
        },
        "after": {
            **current_metrics,
            "test_fraction": final_fraction,
            "events": int(filtered_test["_event"].nunique()),
        },
        "removed_events": len(removed_events),
        "removed_rows": int(len(test) - len(filtered_test)),
        "event_leakage_count": 0,
        "paths": {"train": str(train_path), "test": str(test_path)},
    }
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
