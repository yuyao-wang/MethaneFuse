#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import csv
import json
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


SENSORS = ["s2", "l89", "emit"]
ANGLES = ["0", "90", "360"]
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def norm_path(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return s


def is_pos_label(v):
    s = str(v).strip()
    return s in {"1", "1.0"}


def log(msg):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def check_mask_worker(path):
    try:
        with rasterio.open(path) as ds:
            has_eq1 = False
            has_pos = False
            for b in range(1, ds.count + 1):
                arr = ds.read(b)
                if (not has_eq1) and np.any(arr == 1):
                    has_eq1 = True
                if (not has_pos) and np.any(arr > 0):
                    has_pos = True
                if has_eq1 and has_pos:
                    break
        return path, has_eq1, has_pos, None
    except Exception as e:
        return path, False, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_json", default="plume_check_result.json")
    parser.add_argument("--out_txt", default="plume_check_ids.txt")
    parser.add_argument("--log_every_rows", type=int, default=20000)
    parser.add_argument("--log_every_paths", type=int, default=5000)
    parser.add_argument("--max_examples", type=int, default=30)
    parser.add_argument("--shape_sample_per_sensor", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=20260401)
    parser.add_argument("--mask_workers", type=int, default=8)
    args = parser.parse_args()

    files = [("train", args.train_csv), ("test", args.test_csv)]
    rng = random.Random(args.random_seed)
    t0 = time.time()

    exists_cache = {}
    shape_cache = {}
    shape_err_cache = {}

    def path_exists(p):
        if p in exists_cache:
            return exists_cache[p]
        ok = os.path.exists(p)
        exists_cache[p] = ok
        return ok

    def get_shape(p):
        if p in shape_cache:
            return shape_cache[p]
        if p in shape_err_cache:
            return None
        try:
            with rasterio.open(p) as ds:
                shp = (int(ds.height), int(ds.width))
            shape_cache[p] = shp
            return shp
        except Exception as e:
            shape_err_cache[p] = str(e)
            return None

    issue_names = [
        "missing_plume_path_when_images_present",
        "missing_plume_file_when_images_present",
        "missing_image_file",
        "label1_plume_without_pixel_1",
        "label1_plume_without_any_positive",
        "shape_mismatch",
        "read_error",
    ]
    issue_counts = {k: 0 for k in issue_names}
    issue_ids = {k: set() for k in issue_names}
    issue_ids_by_sensor = {k: defaultdict(set) for k in issue_names}
    issue_examples = {k: [] for k in issue_names}

    def add_issue(name, rid, sensor, example=None):
        issue_counts[name] += 1
        issue_ids[name].add(rid)
        issue_ids_by_sensor[name][sensor].add(rid)
        if example is not None and len(issue_examples[name]) < args.max_examples:
            issue_examples[name].append(example)

    total_rows = 0
    label1_occ_by_plume = defaultdict(list)  # plume -> [(split,id,sensor), ...]
    reservoir = {s: [] for s in SENSORS}
    reservoir_seen = {s: 0 for s in SENSORS}

    log("Phase 1/3: stream CSV and do quick checks ...")
    for split, csv_path in files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                rid = str(row.get("id", ""))
                label = row.get("label", "")
                pos = is_pos_label(label)

                for sensor in SENSORS:
                    img_cols = [f"{sensor}_{a}_path" for a in ANGLES]
                    plume_col = f"{sensor}_plume_path"
                    img_paths = {c: norm_path(row.get(c)) for c in img_cols}
                    plume_path = norm_path(row.get(plume_col))

                    present_img_cols = [c for c, p in img_paths.items() if p is not None]
                    if present_img_cols and plume_path is None:
                        add_issue(
                            "missing_plume_path_when_images_present",
                            rid,
                            sensor,
                            {
                                "split": split,
                                "id": rid,
                                "sensor": sensor,
                                "present_img_cols": present_img_cols,
                            },
                        )
                        continue

                    plume_exists = False
                    if plume_path is not None:
                        plume_exists = path_exists(plume_path)
                        if present_img_cols and (not plume_exists):
                            add_issue(
                                "missing_plume_file_when_images_present",
                                rid,
                                sensor,
                                {
                                    "split": split,
                                    "id": rid,
                                    "sensor": sensor,
                                    "plume_path": plume_path,
                                },
                            )

                    for c, p in img_paths.items():
                        if p is not None and (not path_exists(p)):
                            add_issue(
                                "missing_image_file",
                                rid,
                                sensor,
                                {
                                    "split": split,
                                    "id": rid,
                                    "sensor": sensor,
                                    "image_col": c,
                                    "image_path": p,
                                },
                            )

                    if pos and plume_path and plume_exists:
                        label1_occ_by_plume[plume_path].append((split, rid, sensor))

                    # Reservoir sampling for shape checks
                    if args.shape_sample_per_sensor > 0 and plume_path and plume_exists:
                        for c, p in img_paths.items():
                            if p is None or (not path_exists(p)):
                                continue
                            cand = {
                                "split": split,
                                "id": rid,
                                "sensor": sensor,
                                "image_col": c,
                                "image_path": p,
                                "plume_path": plume_path,
                            }
                            reservoir_seen[sensor] += 1
                            seen = reservoir_seen[sensor]
                            if len(reservoir[sensor]) < args.shape_sample_per_sensor:
                                reservoir[sensor].append(cand)
                            else:
                                j = rng.randint(1, seen)
                                if j <= args.shape_sample_per_sensor:
                                    reservoir[sensor][j - 1] = cand

                if total_rows % args.log_every_rows == 0:
                    log(f"  streamed rows: {total_rows}")

    log(f"  total rows: {total_rows}")
    log(f"  label=1 unique plume paths: {len(label1_occ_by_plume)}")

    log("Phase 2/3: full check label=1 plume has pixel value 1 ...")
    plume_paths = list(label1_occ_by_plume.keys())
    if args.mask_workers <= 1:
        results_iter = (check_mask_worker(p) for p in plume_paths)
    else:
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=args.mask_workers)
        results_iter = ex.map(check_mask_worker, plume_paths, chunksize=64)

    for i, ret in enumerate(results_iter, 1):
        plume_path, has_eq1, has_pos, err = ret
        occs = label1_occ_by_plume[plume_path]
        if err is not None:
            for split, rid, sensor in occs:
                add_issue(
                    "read_error",
                    rid,
                    sensor,
                    {
                        "split": split,
                        "id": rid,
                        "sensor": sensor,
                        "type": "mask_read",
                        "path": plume_path,
                        "error": err,
                    },
                )
            continue

        if not has_eq1:
            for split, rid, sensor in occs:
                add_issue(
                    "label1_plume_without_pixel_1",
                    rid,
                    sensor,
                    {"split": split, "id": rid, "sensor": sensor, "plume_path": plume_path},
                )
        if not has_pos:
            for split, rid, sensor in occs:
                add_issue(
                    "label1_plume_without_any_positive",
                    rid,
                    sensor,
                    {"split": split, "id": rid, "sensor": sensor, "plume_path": plume_path},
                )

        if i % args.log_every_paths == 0:
            log(f"  checked label1 plume paths: {i}/{len(plume_paths)}")

    if args.mask_workers > 1:
        ex.shutdown(wait=True)

    log("Phase 3/3: sampled shape matching checks ...")
    sampled_total = 0
    for sensor in SENSORS:
        for cand in reservoir[sensor]:
            sampled_total += 1
            plume_path = cand["plume_path"]
            img_path = cand["image_path"]
            plume_shape = get_shape(plume_path)
            img_shape = get_shape(img_path)

            if plume_shape is None:
                add_issue(
                    "read_error",
                    cand["id"],
                    sensor,
                    {
                        "split": cand["split"],
                        "id": cand["id"],
                        "sensor": sensor,
                        "type": "plume_shape",
                        "path": plume_path,
                        "error": shape_err_cache.get(plume_path, "unknown"),
                    },
                )
                continue
            if img_shape is None:
                add_issue(
                    "read_error",
                    cand["id"],
                    sensor,
                    {
                        "split": cand["split"],
                        "id": cand["id"],
                        "sensor": sensor,
                        "type": "image_shape",
                        "image_col": cand["image_col"],
                        "path": img_path,
                        "error": shape_err_cache.get(img_path, "unknown"),
                    },
                )
                continue
            if img_shape != plume_shape:
                add_issue(
                    "shape_mismatch",
                    cand["id"],
                    sensor,
                    {
                        "split": cand["split"],
                        "id": cand["id"],
                        "sensor": sensor,
                        "image_col": cand["image_col"],
                        "image_shape": img_shape,
                        "plume_shape": plume_shape,
                        "image_path": img_path,
                        "plume_path": plume_path,
                    },
                )

    id_lists = {k: sorted(v) for k, v in issue_ids.items()}
    id_lists_by_sensor = {
        k: {s: sorted(ids) for s, ids in sens.items()}
        for k, sens in issue_ids_by_sensor.items()
    }

    summary = {
        "total_rows": total_rows,
        "label1_unique_plume_paths_checked": len(label1_occ_by_plume),
        "issue_counts": issue_counts,
        "unique_id_counts": {k: len(v) for k, v in id_lists.items()},
        "shape_sampling": {
            "sample_per_sensor": args.shape_sample_per_sensor,
            "random_seed": args.random_seed,
            "candidates_seen_by_sensor": reservoir_seen,
            "sampled_pairs_by_sensor": {s: len(reservoir[s]) for s in SENSORS},
            "sampled_pairs_total": sampled_total,
        },
        "mask_check": {
            "workers": args.mask_workers,
            "paths_checked": len(plume_paths),
        },
        "timing_sec": {
            "total": round(time.time() - t0, 2),
        },
    }

    out = {
        "summary": summary,
        "id_lists": id_lists,
        "id_lists_by_sensor": id_lists_by_sensor,
        "examples": issue_examples,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(args.out_txt, "w", encoding="utf-8") as f:
        for k, ids in id_lists.items():
            f.write(f"=== {k} ({len(ids)}) ===\n")
            if ids:
                f.write("\n".join(ids) + "\n")
            f.write("\n")

    log("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("out_json:", args.out_json)
    print("out_txt :", args.out_txt)


if __name__ == "__main__":
    main()
