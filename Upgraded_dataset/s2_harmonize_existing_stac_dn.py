#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
FILENAMES = {
    "t0": "s2.tif",
    "prev1": "s2_-7.tif",
    "prev2": "s2_prev2.tif",
    "prev3": "s2_prev3.tif",
    "seasonal": "s2_-90.tif",
    "year": "s2_-360.tif",
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(json.dumps(data, sort_keys=True) + "\n")
    os.replace(temporary, path)


def discover_tasks(
    root: Path,
    desired_dn_add: int,
    table_path: Path,
) -> tuple[list[tuple[Path, Path, dict[str, Any]]], dict[str, int]]:
    tasks: list[tuple[Path, Path, dict[str, Any]]] = []
    counts = {
        "table_rows": 0,
        "stac_candidates": 0,
        "already_harmonized": 0,
        "unexpected_dn_add": 0,
        "missing_tif": 0,
    }
    table = pd.read_csv(table_path, low_memory=False)
    counts["table_rows"] = int(len(table))
    for _, row in table.iterrows():
        plume_id = str(row.get("plume_id", "")).strip()
        if not plume_id:
            continue
        for timepoint in TIMEPOINTS:
            raw_dn_add = row.get(f"{timepoint}_stac_dn_add", 0)
            try:
                current_dn_add = int(float(raw_dn_add or 0))
            except (TypeError, ValueError):
                current_dn_add = 0
            if current_dn_add != 1000:
                continue
            counts["stac_candidates"] += 1
            tif_path = root / timepoint / plume_id / FILENAMES[timepoint]
            sidecar = tif_path.with_name(tif_path.name + ".georef.json")
            metadata: dict[str, Any] = {}
            if sidecar.is_file():
                try:
                    metadata = json.loads(sidecar.read_text())
                except Exception:
                    metadata = {}
            sidecar_dn_add = metadata.get("source_dn_add")
            try:
                if sidecar_dn_add is not None:
                    sidecar_dn_add = int(sidecar_dn_add)
            except (TypeError, ValueError):
                counts["unexpected_dn_add"] += 1
                sidecar_dn_add = None
            if sidecar_dn_add == desired_dn_add:
                counts["already_harmonized"] += 1
                continue
            if sidecar_dn_add not in (None, 1000):
                counts["unexpected_dn_add"] += 1
                continue
            try:
                if not tif_path.is_file() or tif_path.stat().st_size <= 0:
                    counts["missing_tif"] += 1
                    continue
            except OSError:
                counts["missing_tif"] += 1
                continue
            metadata.update(
                {
                    "plume_id": plume_id,
                    "timepoint": timepoint,
                    "source_kind": metadata.get(
                        "source_kind",
                        "stac_cog_resampled",
                    ),
                    "source_dn_add": 1000,
                    "source_item_id": str(
                        row.get(f"{timepoint}_stac_item_id", "")
                    ),
                    "source_product_name": str(
                        row.get(
                            f"{timepoint}_stac_source_product_name",
                            "",
                        )
                    ),
                    "reference_band": (
                        str(
                            row.get(
                                f"{timepoint}_stac_cog_base_url",
                                "",
                            )
                        )
                        + "/B05.tif"
                    ),
                }
            )
            tasks.append((tif_path, sidecar, metadata))
    return tasks, counts


def cache_paths(
    cache_root: Path,
    source_root: Path,
    tif_path: Path,
) -> tuple[Path, Path]:
    relative = tif_path.relative_to(source_root)
    cache_base = cache_root / relative
    corrected = cache_base.with_name(cache_base.name + ".corrected")
    journal = cache_base.with_name(cache_base.name + ".journal.json")
    return corrected, journal


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def correct_one(
    task: tuple[Path, Path, dict[str, Any]],
    source_root: Path,
    cache_root: Path,
    desired_dn_add: int,
) -> dict[str, Any]:
    tif_path, sidecar, metadata = task
    current_dn_add = int(metadata["source_dn_add"])
    delta = desired_dn_add - current_dn_add
    corrected_path, journal_path = cache_paths(cache_root, source_root, tif_path)
    corrected_path.parent.mkdir(parents=True, exist_ok=True)

    journal: dict[str, Any] = {}
    if journal_path.is_file():
        try:
            journal = json.loads(journal_path.read_text())
        except Exception:
            journal = {}

    if (
        journal.get("state") == "corrected_ready"
        and corrected_path.is_file()
        and corrected_path.stat().st_size > 0
    ):
        atomic_copy(corrected_path, tif_path)
    else:
        source_cache = corrected_path.with_name(corrected_path.name + ".source")
        shutil.copyfile(tif_path, source_cache)
        image = tifffile.imread(source_cache)
        if image.dtype != np.uint16:
            raise ValueError(f"unexpected dtype {image.dtype} for {tif_path}")
        if image.ndim != 3 or 12 not in (image.shape[0], image.shape[-1]):
            raise ValueError(f"unexpected shape {image.shape} for {tif_path}")
        adjusted = image.astype(np.int32, copy=True)
        valid = adjusted > 0
        adjusted[valid] += delta
        corrected = np.clip(
            adjusted,
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        tifffile.imwrite(corrected_path, corrected)
        write_json_atomic(
            {
                "state": "corrected_ready",
                "tif_path": str(tif_path),
                "current_dn_add": current_dn_add,
                "desired_dn_add": desired_dn_add,
                "delta": delta,
            },
            journal_path,
        )
        atomic_copy(corrected_path, tif_path)
        source_cache.unlink(missing_ok=True)

    updated = dict(metadata)
    updated["source_dn_add"] = desired_dn_add
    updated["dn_harmonization_previous_add"] = current_dn_add
    updated["dn_harmonization_delta"] = delta
    updated["dn_harmonized_at"] = utc_now()
    write_json_atomic(updated, sidecar)
    corrected_path.unlink(missing_ok=True)
    journal_path.unlink(missing_ok=True)
    return {
        "path": str(tif_path),
        "delta": delta,
        "bytes": int(tif_path.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct already-written STAC COG crops to the same DN range as "
            "COPERNICUS/S2_SR_HARMONIZED, using a local staging cache."
        )
    )
    parser.add_argument(
        "--root",
        default=(
            "/mnt/engg-niulab/yuyao/sensors_raw_data/"
            "S2_point_center_exact_v3"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="/diniuvol/yuyao/s2_dn_harmonize_cache_v4",
    )
    parser.add_argument(
        "--table",
        default=(
            "/home/yuyao/methane_train/Upgrade_data_pipeline/csv/"
            "s2_6time_point_covering_tiles_v3.csv"
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--desired-dn-add", type=int, default=-1000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--report",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_dn_harmonize_v4_report.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    cache_root = Path(args.cache_dir)
    report_path = Path(args.report)
    started = time.monotonic()
    tasks, discovery = discover_tasks(
        root,
        args.desired_dn_add,
        Path(args.table),
    )
    print(
        f"table_rows={discovery['table_rows']} "
        f"stac_candidates={discovery['stac_candidates']} "
        f"to_correct={len(tasks)} "
        f"already={discovery['already_harmonized']} "
        f"unexpected={discovery['unexpected_dn_add']} "
        f"missing_tif={discovery['missing_tif']}",
        flush=True,
    )
    if args.dry_run:
        return 0

    cache_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    corrected_bytes = 0
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                correct_one,
                task,
                root,
                cache_root,
                args.desired_dn_add,
            ): task[0]
            for task in tasks
        }
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                result = future.result()
                corrected_bytes += int(result["bytes"])
            except Exception as exc:
                failures.append(
                    {
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if (
                completed % max(1, args.progress_every) == 0
                or completed == len(tasks)
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = completed / elapsed
                remaining = (len(tasks) - completed) / max(rate, 1e-6)
                print(
                    f"progress={completed}/{len(tasks)} "
                    f"failed={len(failures)} rate={rate:.2f}/s "
                    f"eta_min={remaining / 60:.1f}",
                    flush=True,
                )

    report = {
        "root": str(root),
        "cache_dir": str(cache_root),
        "desired_dn_add": int(args.desired_dn_add),
        "discovery": discovery,
        "tasks": len(tasks),
        "completed": completed - len(failures),
        "failures": failures,
        "corrected_bytes": corrected_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "finished_at": utc_now(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
