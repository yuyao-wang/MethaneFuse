#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


PATH_COLUMNS = (
    "s2_0_std_512",
    "s2_-7_std_512",
    "s2_prev2_std_512",
    "s2_prev3_std_512",
    "s2_-90_std_512",
    "s2_-360_std_512",
)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checksum(array: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(array).view(np.uint8))


def inspect_target(path_text: str) -> dict[str, str]:
    path = Path(path_text)
    sidecar = path.with_name(path.name + ".georef.json")
    try:
        if not sidecar.is_file():
            return {"path": path_text, "status": "preexisting_correct"}
        metadata = json.loads(sidecar.read_text())
        if metadata.get("canonical_band_order_complete"):
            status = "already_complete"
        elif metadata.get("b8a_b09_fill_complete"):
            status = "needs_refetch"
        elif metadata.get("b08_b09_fill_complete"):
            status = "would_fix"
        else:
            status = "preexisting_correct"
        return {"path": path_text, "status": status}
    except Exception as exc:
        return {
            "path": path_text,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def fix_target(path_text: str) -> dict[str, str]:
    path = Path(path_text)
    sidecar = path.with_name(path.name + ".georef.json")
    try:
        if not sidecar.is_file():
            return {"path": path_text, "status": "preexisting_correct"}
        metadata = json.loads(sidecar.read_text())
        if metadata.get("canonical_band_order_complete"):
            return {"path": path_text, "status": "already_complete"}
        if metadata.get("b8a_b09_fill_complete"):
            return {"path": path_text, "status": "needs_refetch"}
        if not metadata.get("b08_b09_fill_complete"):
            return {"path": path_text, "status": "preexisting_correct"}

        image = tifffile.memmap(path, mode="r+")
        if image.shape != (12, 512, 512):
            return {
                "path": path_text,
                "status": "failed",
                "error": f"unexpected_shape={image.shape}",
            }
        band7_crc = checksum(image[7])
        band8_crc = checksum(image[8])
        repair = metadata.get("canonical_band_order_repair")
        if isinstance(repair, dict) and repair.get("state") == "in_progress":
            before7 = int(repair["before_index7_crc32"])
            before8 = int(repair["before_index8_crc32"])
            if (band7_crc, band8_crc) == (before8, before7):
                metadata["canonical_band_order_complete"] = True
                metadata["canonical_band_order_repair"]["state"] = "complete"
                atomic_write_json(sidecar, metadata)
                return {"path": path_text, "status": "recovered_complete"}
            if (band7_crc, band8_crc) != (before7, before8):
                return {
                    "path": path_text,
                    "status": "failed",
                    "error": "in_progress_crc_mismatch",
                }
        else:
            metadata["canonical_band_order_repair"] = {
                "state": "in_progress",
                "operation": "swap_index7_index8",
                "before_index7_band": "B8A",
                "before_index8_band": "B08",
                "before_index7_crc32": band7_crc,
                "before_index8_crc32": band8_crc,
            }
            atomic_write_json(sidecar, metadata)

        temporary = np.array(image[7], copy=True)
        image[7] = image[8]
        image[8] = temporary
        image.flush()
        after7_crc = checksum(image[7])
        after8_crc = checksum(image[8])
        del image
        repair = metadata["canonical_band_order_repair"]
        if (
            after7_crc != int(repair["before_index8_crc32"])
            or after8_crc != int(repair["before_index7_crc32"])
        ):
            return {
                "path": path_text,
                "status": "failed",
                "error": "post_swap_crc_mismatch",
            }
        repair["state"] = "complete"
        repair["after_index7_band"] = "B08"
        repair["after_index8_band"] = "B8A"
        repair["after_index7_crc32"] = after7_crc
        repair["after_index8_crc32"] = after8_crc
        metadata["canonical_band_order_complete"] = True
        atomic_write_json(sidecar, metadata)
        return {"path": path_text, "status": "fixed"}
    except Exception as exc:
        return {
            "path": path_text,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets: set[str] = set()
    for value in args.manifests.split(","):
        frame = pd.read_csv(value.strip(), low_memory=False)
        for column in PATH_COLUMNS:
            targets.update(frame[column].dropna().astype(str))
    ordered_targets = sorted(targets)
    started = time.monotonic()
    records: list[dict[str, str]] = []
    if args.dry_run:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            records = list(pool.map(inspect_target, ordered_targets))
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for index, result in enumerate(
                pool.map(fix_target, ordered_targets),
                start=1,
            ):
                records.append(result)
                if (
                    index % max(1, args.progress_every) == 0
                    or index == len(ordered_targets)
                ):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    counts = pd.Series(
                        [record["status"] for record in records]
                    ).value_counts()
                    print(
                        f"[OrderFix] {index}/{len(ordered_targets)} "
                        f"rate={index / elapsed:.1f}/s "
                        f"fixed={int(counts.get('fixed', 0))} "
                        f"failed={int(counts.get('failed', 0))}",
                        flush=True,
                    )
    counts = (
        pd.Series([record["status"] for record in records])
        .value_counts()
        .sort_index()
        .to_dict()
    )
    failures = [record for record in records if record["status"] == "failed"]
    needs_refetch = [
        record["path"] for record in records if record["status"] == "needs_refetch"
    ]
    report = {
        "targets": len(ordered_targets),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "needs_refetch": needs_refetch,
        "failures": failures[:100],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
