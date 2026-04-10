#!/usr/bin/env python3
"""Prepare Compute Canada data migration plan from CSV manifests.

This script:
1) extracts all non-empty absolute path values from columns containing "path";
2) checks source file sizes (best effort);
3) assigns files to destination A first, then destination B by capacity;
4) writes rsync file lists for A/B and rewritten *.cc.csv manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


csv.field_size_limit(1024 * 1024 * 1024)


@dataclass(frozen=True)
class FileMeta:
    path: str
    size: int
    exists: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CC transfer plan and rewritten manifests.")
    p.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        required=True,
        help="Input CSV manifest (repeat for train/test).",
    )
    p.add_argument(
        "--dest-a",
        default="/project/def-juliana2/panopticon_data_mirror",
        help="Primary destination root (priority target).",
    )
    p.add_argument(
        "--dest-b",
        default="/project/def-iamniudi/panopticon_data_mirror",
        help="Secondary destination root (overflow target).",
    )
    p.add_argument(
        "--cap-a-gb",
        type=float,
        default=930.0,
        help="Soft capacity for destination A in GiB.",
    )
    p.add_argument(
        "--cap-b-gb",
        type=float,
        default=930.0,
        help="Soft capacity for destination B in GiB.",
    )
    p.add_argument(
        "--out-dir",
        default="cc_migration_plan",
        help="Output directory for lists, mapping, and rewritten CSVs.",
    )
    p.add_argument(
        "--host",
        default="yuyao16@narval.alliancecan.ca",
        help="SSH target printed in transfer command templates.",
    )
    p.add_argument(
        "--no-size-probe",
        action="store_true",
        help="Skip os.stat() on each source file. Faster but capacity split is approximate.",
    )
    p.add_argument(
        "--unknown-size-mib",
        type=float,
        default=16.0,
        help="Estimated size (MiB) used when size probe is disabled or file is missing.",
    )
    return p.parse_args()


def _is_path_column(col: str) -> bool:
    return "path" in col.lower()


def _iter_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def collect_paths(csv_paths: Sequence[Path]) -> Dict[str, List[str]]:
    path_columns_by_csv: Dict[str, List[str]] = {}
    all_paths: Dict[str, None] = {}

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")
            path_cols = [c for c in reader.fieldnames if _is_path_column(c)]
            path_columns_by_csv[str(csv_path)] = path_cols
            for row in reader:
                for col in path_cols:
                    val = (row.get(col) or "").strip()
                    if not val:
                        continue
                    if val.lower() in {"nan", "none", "null"}:
                        continue
                    if not val.startswith("/"):
                        continue
                    all_paths[val] = None

    return {
        "path_columns_by_csv": path_columns_by_csv,  # type: ignore[return-value]
        "all_paths": sorted(all_paths.keys()),  # type: ignore[return-value]
    }


def probe_files(paths: Sequence[str], *, do_probe: bool) -> List[FileMeta]:
    out: List[FileMeta] = []
    if not do_probe:
        for p in paths:
            out.append(FileMeta(path=p, size=0, exists=True))
        return out

    for p in paths:
        try:
            st = os.stat(p)
            out.append(FileMeta(path=p, size=int(st.st_size), exists=True))
        except OSError:
            out.append(FileMeta(path=p, size=0, exists=False))
    return out


def assign_files(
    files: Sequence[FileMeta],
    cap_a_gib: float,
    cap_b_gib: float,
    unknown_size_bytes: int,
) -> Dict[str, str]:
    cap_a = int(cap_a_gib * (1024**3))
    cap_b = int(cap_b_gib * (1024**3))
    used_a = 0
    used_b = 0
    assignment: Dict[str, str] = {}

    # Large-first greedy packing keeps A closer to capacity while minimizing overflow.
    ranked = sorted(files, key=lambda x: x.size, reverse=True)
    for fm in ranked:
        sz = max(0, int(fm.size))
        if sz == 0:
            sz = unknown_size_bytes
        if used_a + sz <= cap_a:
            assignment[fm.path] = "A"
            used_a += sz
        else:
            assignment[fm.path] = "B"
            used_b += sz

    # If B is above soft cap, keep assignment but report it in summary.
    assignment["__used_a_bytes__"] = str(used_a)
    assignment["__used_b_bytes__"] = str(used_b)
    assignment["__cap_a_bytes__"] = str(cap_a)
    assignment["__cap_b_bytes__"] = str(cap_b)
    return assignment


def make_new_path(old_path: str, dest_root: Path) -> str:
    return str((dest_root / old_path.lstrip("/")).resolve())


def rewrite_csv(
    src_csv: Path,
    dst_csv: Path,
    path_cols: Sequence[str],
    path_map: Dict[str, str],
) -> None:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with src_csv.open("r", newline="", encoding="utf-8") as fin, dst_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {src_csv}")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            for c in path_cols:
                v = (row.get(c) or "").strip()
                if v in path_map:
                    row[c] = path_map[v]
            writer.writerow(row)


def gib(n_bytes: int) -> float:
    return float(n_bytes) / float(1024**3)


def main() -> None:
    args = parse_args()
    csv_paths = [Path(p).expanduser().resolve() for p in args.csv_paths]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_a = Path(args.dest_a).expanduser()
    dest_b = Path(args.dest_b).expanduser()

    data = collect_paths(csv_paths)
    path_columns_by_csv: Dict[str, List[str]] = data["path_columns_by_csv"]  # type: ignore[assignment]
    all_paths: List[str] = data["all_paths"]  # type: ignore[assignment]
    files = probe_files(all_paths, do_probe=(not args.no_size_probe))
    unknown_size_bytes = int(max(1.0, args.unknown_size_mib) * 1024 * 1024)

    assignment = assign_files(
        files,
        cap_a_gib=args.cap_a_gb,
        cap_b_gib=args.cap_b_gb,
        unknown_size_bytes=unknown_size_bytes,
    )
    used_a = int(assignment.pop("__used_a_bytes__"))
    used_b = int(assignment.pop("__used_b_bytes__"))
    cap_a = int(assignment.pop("__cap_a_bytes__"))
    cap_b = int(assignment.pop("__cap_b_bytes__"))

    list_a = out_dir / "rsync_files_A.txt"
    list_b = out_dir / "rsync_files_B.txt"
    missing = out_dir / "missing_files.txt"
    map_tsv = out_dir / "path_mapping.tsv"

    rel_a: List[str] = []
    rel_b: List[str] = []
    missing_paths: List[str] = []
    path_map: Dict[str, str] = {}

    for fm in files:
        bucket = assignment[fm.path]
        if bucket == "A":
            rel_a.append(fm.path.lstrip("/"))
            new_path = make_new_path(fm.path, dest_a)
        else:
            rel_b.append(fm.path.lstrip("/"))
            new_path = make_new_path(fm.path, dest_b)
        path_map[fm.path] = new_path
        if not fm.exists and (not args.no_size_probe):
            missing_paths.append(fm.path)

    list_a.write_text("\n".join(rel_a) + ("\n" if rel_a else ""), encoding="utf-8")
    list_b.write_text("\n".join(rel_b) + ("\n" if rel_b else ""), encoding="utf-8")
    missing.write_text("\n".join(missing_paths) + ("\n" if missing_paths else ""), encoding="utf-8")

    with map_tsv.open("w", encoding="utf-8", newline="") as f:
        f.write("old_path\tnew_path\tbucket\n")
        for old in sorted(path_map.keys()):
            bucket = assignment[old]
            f.write(f"{old}\t{path_map[old]}\t{bucket}\n")

    rewritten_csvs: List[str] = []
    for src in csv_paths:
        dst = out_dir / (src.stem + ".cc.csv")
        rewrite_csv(
            src_csv=src,
            dst_csv=dst,
            path_cols=path_columns_by_csv[str(src)],
            path_map=path_map,
        )
        rewritten_csvs.append(str(dst))

    summary = {
        "csv_inputs": [str(p) for p in csv_paths],
        "unique_paths": len(files),
        "missing_files": len(missing_paths),
        "size_probe_enabled": not args.no_size_probe,
        "unknown_size_mib": args.unknown_size_mib,
        "assigned_A_files": len(rel_a),
        "assigned_B_files": len(rel_b),
        "used_A_gib": round(gib(used_a), 3),
        "used_B_gib": round(gib(used_b), 3),
        "cap_A_gib": round(gib(cap_a), 3),
        "cap_B_gib": round(gib(cap_b), 3),
        "rewritten_csvs": rewritten_csvs,
        "rsync_list_A": str(list_a),
        "rsync_list_B": str(list_b),
        "missing_list": str(missing),
        "mapping_tsv": str(map_tsv),
        "dest_A": str(dest_a),
        "dest_B": str(dest_b),
        "commands": {
            "create_dest_dirs": (
                f"ssh {args.host} 'mkdir -p {dest_a} {dest_b}'"
            ),
            "rsync_A": (
                f"rsync -a --partial --append-verify --info=progress2 "
                f"--files-from='{list_a}' / {args.host}:'{dest_a}/'"
            ),
            "rsync_B": (
                f"rsync -a --partial --append-verify --info=progress2 "
                f"--files-from='{list_b}' / {args.host}:'{dest_b}/'"
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] wrote: {out_dir}")
    print(f"[OK] unique paths: {len(files)}")
    print(f"[OK] missing files: {len(missing_paths)}")
    print(f"[OK] A usage: {gib(used_a):.2f} GiB / {gib(cap_a):.2f} GiB")
    print(f"[OK] B usage: {gib(used_b):.2f} GiB / {gib(cap_b):.2f} GiB")
    print("[NEXT] open summary:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
