#!/usr/bin/env python3
"""Fail-closed audit of the immutable 3,072-row clean-L89 GPU0 smoke pair."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from research.pretraining_20260727 import l89_ragged_cls_experiment as cache


FORBIDDEN_RE = re.compile(
    r"(^|[._/\\-])(test|sealed|holdout|outer)([._/\\-]|$)", re.IGNORECASE
)
FROZEN_WEIGHTS_SHA256 = (
    "55024f411a7f383ed1a646d9b833b65683a4443846603fc7d37591d5afe9d26e"
)
FROZEN_E2E_RECEIPT_SHA256 = (
    "17d9a77257d3f78856ffa48b052dcc70646f5d8d40637ba882b93cf335bb05bc"
)
FROZEN_ARTIFACTS = {
    "train": {
        "cache_sha256": (
            "0a5ce87b631da3ac8911a768e8dcf0378e3c1d2c96a45bbd0279990ae33d9d4a"
        ),
        "csv_sha256": (
            "eae72baa9d16485e3dd3ff9d9eb2edba32271f12c2df54a4af4cd9d44c2aa73c"
        ),
    },
    "val": {
        "cache_sha256": (
            "6bf5323155994244a6a95edf09494401b001cdff287adcd3a15845ea886f7c4d"
        ),
        "csv_sha256": (
            "b83f2160c81505db7552acd962bf3268f4d26413aa54e740117ed992ad5d4d53"
        ),
    },
}


def safe_file(value: str | Path, *, purpose: str) -> Path:
    path = Path(value).expanduser().resolve()
    if FORBIDDEN_RE.search(str(path)):
        raise ValueError(f"{purpose} contains a held-out marker: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audit_one(
    *,
    split: str,
    cache_path: Path,
    csv_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    frozen = FROZEN_ARTIFACTS[split]
    observed_cache_sha = cache.sha256_file(cache_path)
    observed_csv_sha = cache.sha256_file(csv_path)
    if observed_cache_sha != frozen["cache_sha256"]:
        raise ValueError(f"{split} smoke cache SHA changed.")
    if observed_csv_sha != frozen["csv_sha256"]:
        raise ValueError(f"{split} smoke CSV SHA changed.")
    payload = cache.torch_load_trusted(cache_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{split} smoke cache is not a mapping.")
    cache.validate_cache_payload(
        payload, path=cache_path, expected_split=split
    )
    if int(payload["features"].shape[0]) != 3072:
        raise ValueError(f"{split} smoke cache is not exactly 3,072 rows.")
    if tuple(payload["features"].shape[1:]) != (6, 768):
        raise ValueError(f"{split} smoke feature shape is not [3072,6,768].")
    if payload.get("weights_sha256") != FROZEN_WEIGHTS_SHA256:
        raise ValueError(f"{split} smoke base-weight SHA mismatch.")
    if payload.get("csv_sha256") != observed_csv_sha:
        raise ValueError(f"{split} smoke payload/CSV SHA mismatch.")
    summary_path = cache_path.with_suffix(".pt.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        int(summary.get("rows", -1)) != 3072
        or summary.get("split") != split
        or int(summary.get("read_errors", -1)) != 0
        or int(summary.get("invalid_t0_rows", -1)) != 0
        or summary.get("weights_sha256") != FROZEN_WEIGHTS_SHA256
        or summary.get("csv_sha256") != observed_csv_sha
        or summary.get("cache_sha256") != observed_cache_sha
    ):
        raise ValueError(f"{split} smoke JSON receipt is inconsistent.")
    return payload, {
        "split": split,
        "rows": 3072,
        "timepoints": 6,
        "feature_dim": 768,
        "cache": str(cache_path),
        "cache_sha256": observed_cache_sha,
        "csv": str(csv_path),
        "csv_sha256": observed_csv_sha,
        "summary": str(summary_path),
        "summary_sha256": cache.sha256_file(summary_path),
        "read_errors": 0,
        "invalid_t0_rows": 0,
        "weights_sha256": FROZEN_WEIGHTS_SHA256,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    e2e_receipt_path = safe_file(
        args.e2e_receipt, purpose="same-GPU end-to-end smoke receipt"
    )
    if cache.sha256_file(e2e_receipt_path) != FROZEN_E2E_RECEIPT_SHA256:
        raise ValueError("Same-GPU end-to-end smoke receipt SHA changed.")
    e2e = json.loads(e2e_receipt_path.read_text(encoding="utf-8"))
    if (
        e2e.get("schema_version") != "l89-clean-e2e-cache-smoke3072-v1"
        or e2e.get("runtime", {}).get("physical_gpu") != 0
        or e2e.get("runtime", {}).get("both_processes_concurrent") is not True
        or int(e2e.get("runtime", {}).get("batch_size_per_process", -1)) != 64
        or int(e2e.get("runtime", {}).get("num_workers_per_process", -1)) != 6
        or int(e2e.get("runtime", {}).get("prefetch_factor", -1)) != 1
        or e2e.get("weights_sha256") != FROZEN_WEIGHTS_SHA256
    ):
        raise ValueError("Same-GPU end-to-end smoke receipt contract changed.")
    weights = safe_file(args.base_weights, purpose="frozen base weights")
    if cache.sha256_file(weights) != FROZEN_WEIGHTS_SHA256:
        raise ValueError("Frozen base-weight file SHA changed.")
    train_cache = safe_file(args.train_cache, purpose="train smoke cache")
    val_cache = safe_file(args.val_cache, purpose="validation smoke cache")
    train_csv = safe_file(args.train_csv, purpose="train smoke CSV")
    val_csv = safe_file(args.val_csv, purpose="validation smoke CSV")
    for name, record, expected_cache, expected_csv in (
        ("train", e2e["train"], train_cache, train_csv),
        ("development", e2e["development"], val_cache, val_csv),
    ):
        recorded_cache = safe_file(
            record["cache_path"], purpose=f"receipt {name} smoke cache"
        )
        recorded_csv = safe_file(
            record["manifest_path"], purpose=f"receipt {name} smoke CSV"
        )
        frozen_key = "train" if name == "train" else "val"
        if (
            recorded_cache != expected_cache
            or recorded_csv != expected_csv
            or record.get("cache_sha256")
            != FROZEN_ARTIFACTS[frozen_key]["cache_sha256"]
            or record.get("manifest_sha256")
            != FROZEN_ARTIFACTS[frozen_key]["csv_sha256"]
            or int(record.get("rows", -1)) != 3072
            or int(record.get("read_errors", -1)) != 0
            or int(record.get("invalid_t0_rows", -1)) != 0
        ):
            raise ValueError(f"End-to-end receipt {name} evidence mismatch.")
    train, train_receipt = audit_one(
        split="train", cache_path=train_cache, csv_path=train_csv
    )
    val, val_receipt = audit_one(
        split="val", cache_path=val_cache, csv_path=val_csv
    )
    _, _, pair = cache.load_cache_pair(train_cache, val_cache)
    if int(pair["event_overlap"]) != 0:
        raise ValueError("Smoke train/validation canonical events overlap.")
    output = {
        "schema_version": "l89-clean-smoke3072-prerequisite-v1",
        "status": "validated",
        "train": train_receipt,
        "validation": val_receipt,
        "pair_audit": pair,
        "same_gpu_end_to_end_receipt": {
            "path": str(e2e_receipt_path),
            "sha256": FROZEN_E2E_RECEIPT_SHA256,
            "combined_gpu_memory_mib": int(
                e2e["runtime"]["observed_combined_gpu_memory_mib"]
            ),
            "remaining_gpu_memory_mib": int(
                e2e["runtime"]["observed_remaining_gpu_memory_mib"]
            ),
        },
        "ordered_identity_payloads_loaded": bool(
            train["ids"] and val["ids"]
        ),
        "gpu_protocol": {
            "physical_gpu": 0,
            "concurrent_processes": 2,
            "batch_size_per_process": 64,
        },
        "test_or_sealed_or_holdout_or_outer_read": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    root = Path(
        "/diniuvol/yuyao/methanefuse_research_20260728/"
        "l89_clean_prep_v1/smoke3072"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default=str(root / "base_train.pt"))
    parser.add_argument("--val-cache", default=str(root / "base_val.pt"))
    parser.add_argument("--train-csv", default=str(root / "manifests/train.csv"))
    parser.add_argument("--val-csv", default=str(root / "manifests/dev.csv"))
    parser.add_argument(
        "--base-weights",
        default="/home/yuyao/panopticon/weights/panopticon_vitb14_teacher.pth",
    )
    parser.add_argument(
        "--e2e-receipt",
        default=str(
            Path(__file__).resolve().with_name(
                "l89_clean_e2e_smoke3072_v1.json"
            )
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
