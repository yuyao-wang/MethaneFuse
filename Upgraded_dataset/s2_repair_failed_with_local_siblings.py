#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd


TIMEPOINTS = ("t0", "prev1", "prev2", "prev3", "seasonal", "year")
DEFAULT_ROOTS = ",".join(
    [
        "/diniuvol/yuyao/s2_cdse_point_repair_cache",
        "/diniuvol/yuyao/s2_early_boundary_products",
        "/mnt/engg-niulab/yuyao/sensors_raw_data/S2/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_S2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_s2_90360",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2",
        "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7",
    ]
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def load_resolver(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "s2_local_tile_resolver",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_local_sibling_lookup(
    resolver: Any,
    roots: tuple[Path, ...],
) -> tuple[
    Callable[[dict[str, str], tuple[Path, ...]], list[dict[str, Any]]],
    dict[str, int],
]:
    exact_index: dict[tuple[str, ...], dict[str, Path]] = {}
    acquisition_index: dict[tuple[str, ...], dict[str, Path]] = {}
    root_counts: dict[str, int] = {}
    for root in roots:
        try:
            names = os.listdir(root)
        except OSError:
            root_counts[str(root)] = 0
            continue
        count = 0
        for name in names:
            if not name.endswith(".SAFE"):
                continue
            try:
                parsed = resolver.parse_product(name)
            except ValueError:
                continue
            count += 1
            path = root / name
            exact_key = (
                parsed["satellite"],
                parsed["sensing"],
                parsed["baseline"],
                parsed["orbit"],
                parsed["generation"],
            )
            acquisition_key = (
                parsed["satellite"],
                parsed["sensing"],
                parsed["orbit"],
            )
            exact_index.setdefault(exact_key, {}).setdefault(name, path)
            acquisition_index.setdefault(acquisition_key, {}).setdefault(
                name,
                path,
            )
        root_counts[str(root)] = count

    grid_cache: dict[str, Any] = {}

    def lookup(
        wanted: dict[str, str],
        _: tuple[Path, ...],
    ) -> list[dict[str, Any]]:
        exact_key = (
            wanted["satellite"],
            wanted["sensing"],
            wanted["baseline"],
            wanted["orbit"],
            wanted["generation"],
        )
        acquisition_key = (
            wanted["satellite"],
            wanted["sensing"],
            wanted["orbit"],
        )
        products = dict(acquisition_index.get(acquisition_key, {}))
        products.update(exact_index.get(exact_key, {}))
        siblings: list[dict[str, Any]] = []
        for name, path in products.items():
            grid = grid_cache.get(name)
            if grid is None:
                try:
                    grid = resolver.local_grid(path)
                except (OSError, ValueError, FileNotFoundError):
                    continue
                grid_cache[name] = grid
            siblings.append(
                {
                    "product_name": name,
                    "product_dir": str(path),
                    "grid": grid,
                }
            )
        return siblings

    return lookup, root_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair failed point crops using only locally available sibling "
            "SAFE tiles from the same Sentinel-2 acquisition."
        )
    )
    parser.add_argument(
        "--progress-csv",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_recrop_progress_v4.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_recrop_retry_local_v4.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_canonical_6510_recrop_retry_local_v4_report.json"
        ),
    )
    parser.add_argument(
        "--resolver",
        default=(
            "/home/yuyao/panopticon/Upgraded_dataset/"
            "s2_resolve_point_covering_tiles.py"
        ),
    )
    parser.add_argument("--product-roots", default=DEFAULT_ROOTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolver = load_resolver(Path(args.resolver))
    roots = tuple(
        Path(value.strip())
        for value in args.product_roots.split(",")
        if value.strip()
    )
    sibling_lookup, root_counts = build_local_sibling_lookup(
        resolver,
        roots,
    )
    resolver.local_sibling_products = sibling_lookup
    frame = pd.read_csv(args.progress_csv, low_memory=False)
    groups: dict[str, list[dict[str, Any]]] = {}
    failed_keys: set[tuple[int, str]] = set()
    for index, row in frame.iterrows():
        for timepoint in TIMEPOINTS:
            if clean(row.get(f"{timepoint}_recrop_status")) != "failed":
                continue
            product_name = clean(row.get(f"{timepoint}_product_name"))
            product_id = clean(row.get(f"{timepoint}_product_id"))
            if not product_name or not product_id:
                continue
            task = {
                "row_index": int(index),
                "plume_id": clean(row.get("plume_id")),
                "timepoint": timepoint,
                "product_name": product_name,
                "product_id": product_id,
                "longitude": float(row["plume_longitude"]),
                "latitude": float(row["plume_latitude"]),
            }
            groups.setdefault(product_name, []).append(task)
            failed_keys.add((int(index), timepoint))

    repaired_records: list[dict[str, Any]] = []
    unresolved_records: list[dict[str, Any]] = []
    for product_name, tasks in groups.items():
        wanted = resolver.parse_product(product_name)
        repaired, unresolved = resolver.repair_with_local_siblings(
            wanted,
            tasks,
            roots,
        )
        repaired_records.extend(repaired)
        unresolved_records.extend(unresolved)

    repaired_keys: set[tuple[int, str]] = set()
    for record in repaired_records:
        index = int(record["row_index"])
        timepoint = record["timepoint"]
        repaired_keys.add((index, timepoint))
        selected_name = clean(record["selected_product_name"])
        frame.at[index, f"{timepoint}_product_name"] = selected_name
        frame.at[index, f"{timepoint}_product_id"] = clean(
            record["selected_product_id"]
        )
        frame.at[index, f"{timepoint}_local_mosaic_json"] = clean(
            record.get("local_mosaic_json")
        )
        local_product = resolver.local_product_dir(
            selected_name,
            roots,
        )
        frame.at[index, f"{timepoint}_product_local_path"] = (
            str(local_product) if local_product is not None else ""
        )
        frame.at[index, f"{timepoint}_recrop_status"] = ""
        frame.at[index, f"{timepoint}_recrop_message"] = ""
        frame.at[index, f"{timepoint}_tile_repair_status"] = clean(
            record.get("tile_repair_status")
        )
        frame.at[index, f"{timepoint}_tile_margin_pixels"] = record.get(
            "tile_margin_pixels",
            "",
        )
        frame.at[index, f"{timepoint}_center_offset_pixels"] = record.get(
            "center_offset_pixels",
            "",
        )

    for index, timepoint in failed_keys - repaired_keys:
        frame.at[index, f"{timepoint}_product_name"] = ""
        frame.at[index, f"{timepoint}_product_id"] = ""
        frame.at[index, f"{timepoint}_product_local_path"] = ""

    output_path = Path(args.output_csv)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}"
    )
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    report = {
        "failed_tasks": len(failed_keys),
        "failed_product_groups": len(groups),
        "locally_repaired_tasks": len(repaired_records),
        "unresolved_tasks": len(unresolved_records),
        "repair_status_counts": pd.Series(
            [
                clean(record.get("tile_repair_status"))
                for record in repaired_records
            ]
        )
        .value_counts()
        .to_dict(),
        "product_root_counts": root_counts,
        "unresolved_examples": unresolved_records[:20],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
