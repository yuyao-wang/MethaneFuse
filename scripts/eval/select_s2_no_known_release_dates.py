#!/usr/bin/env python3
"""Select reproducible Sentinel-2 acquisitions outside release campaigns.

The output labels are deliberately described as ``no_known_controlled_release``:
being outside a published campaign excludes the campaign release, but is not
proof that no unrelated methane source was present in the image.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd


COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-csv", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cloud-pct", type=float, default=20.0)
    parser.add_argument("--ee-project", default="")
    parser.add_argument("--authenticate", action="store_true")
    return parser.parse_args()


def utc_date(value: object) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def iso_z(milliseconds: int) -> str:
    return (
        datetime.fromtimestamp(milliseconds / 1000.0, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def excluded_dates(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        for column in ("datetime", "event_time", "acquisition_time_utc"):
            if column in frame.columns:
                result.update(filter(None, (utc_date(value) for value in frame[column])))
                break
    return result


def initialize_ee(args: argparse.Namespace) -> None:
    if args.authenticate:
        ee.Authenticate()
    kwargs = {"project": args.ee_project} if args.ee_project else {}
    ee.Initialize(**kwargs)


def site_acquisitions(row: pd.Series, cloud_pct: float) -> list[dict]:
    point = ee.Geometry.Point(float(row.longitude), float(row.latitude))
    collection = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(point)
        .filterDate(str(row.query_start), str(row.query_end))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .sort("system:time_start")
    )
    payload = collection.reduceColumns(
        ee.Reducer.toList(3),
        ["system:time_start", "system:index", "CLOUDY_PIXEL_PERCENTAGE"],
    ).get("list").getInfo()
    deduplicated: dict[int, dict] = {}
    for milliseconds, image_id, cloud in payload or []:
        milliseconds = int(milliseconds)
        candidate = {
            "milliseconds": milliseconds,
            "selected_id": str(image_id),
            "cloud_pct": float(cloud),
        }
        previous = deduplicated.get(milliseconds)
        if previous is None or candidate["cloud_pct"] < previous["cloud_pct"]:
            deduplicated[milliseconds] = candidate
    return list(deduplicated.values())


def main() -> None:
    args = parse_args()
    initialize_ee(args)
    sites = pd.read_csv(args.sites_csv, low_memory=False)
    required = {
        "site_id",
        "latitude",
        "longitude",
        "query_start",
        "query_end",
        "campaign_start",
        "campaign_end",
        "n_select",
    }
    missing = sorted(required - set(sites.columns))
    if missing:
        raise ValueError(f"sites CSV missing columns: {missing}")
    excluded = excluded_dates(args.exclude_csv)
    rng = random.Random(args.seed)
    records: list[dict] = []
    for _, site in sites.iterrows():
        campaign_start = pd.Timestamp(site.campaign_start, tz="UTC")
        campaign_end = pd.Timestamp(site.campaign_end, tz="UTC")
        eligible = []
        for candidate in site_acquisitions(site, args.cloud_pct):
            timestamp = pd.Timestamp(candidate["milliseconds"], unit="ms", tz="UTC")
            date = timestamp.strftime("%Y-%m-%d")
            if campaign_start.normalize() <= timestamp.normalize() <= campaign_end.normalize():
                continue
            if date in excluded:
                continue
            eligible.append(candidate)
        count = int(site.n_select)
        if len(eligible) < count:
            raise ValueError(
                f"{site.site_id}: only {len(eligible)} eligible acquisitions for {count} requested"
            )
        selected = sorted(rng.sample(eligible, count), key=lambda item: item["milliseconds"])
        for candidate in selected:
            time = iso_z(candidate["milliseconds"])
            compact = time.replace("-", "").replace(":", "").replace("T", "t").replace("Z", "")
            plume_id = f"temporal_neg_{site.site_id}_{compact}"
            records.append(
                {
                    "plume_id": plume_id,
                    "site_id": str(site.site_id),
                    "plume_latitude": float(site.latitude),
                    "plume_longitude": float(site.longitude),
                    "event_time": time,
                    "datetime": time,
                    "t0_image_time": time,
                    "label": 0,
                    "selected_id": candidate["selected_id"],
                    "cloud_pct": candidate["cloud_pct"],
                    "negative_evidence": "outside_published_controlled_release_campaign; no_known_controlled_release",
                    "campaign_start": str(site.campaign_start),
                    "campaign_end": str(site.campaign_end),
                    "selection_seed": int(args.seed),
                }
            )
    output = pd.DataFrame.from_records(records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(output.to_string(index=False))
    print({"output": str(args.output_csv), "rows": len(output), "seed": args.seed})


if __name__ == "__main__":
    main()
