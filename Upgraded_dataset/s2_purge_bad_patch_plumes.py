#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


def sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "unknown"


def rewrite_manifest(path: Path, bad_plume_ids: set[str]) -> tuple[int, int]:
    frame = pd.read_csv(path, low_memory=False)
    before = len(frame)
    kept = frame[~frame["plume_id"].astype(str).isin(bad_plume_ids)].copy()
    temporary = path.with_suffix(path.suffix + ".part")
    kept.to_csv(temporary, index=False)
    temporary.replace(path)
    return before, len(kept)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit_json).read_text())
    manifests = {
        "train": Path(args.train_csv),
        "test": Path(args.test_csv),
    }
    patch_root = Path(args.patch_root)
    report: dict[str, object] = {"splits": {}}

    for split_name, manifest in manifests.items():
        bad = {
            str(value)
            for value in audit.get("bad_plume_ids_by_split", {}).get(split_name, [])
        }
        before, after = rewrite_manifest(manifest, bad)
        removed_dirs = 0
        for plume_id in bad:
            plume_dir = patch_root / split_name / sanitize_component(plume_id)
            if plume_dir.exists():
                shutil.rmtree(plume_dir)
                removed_dirs += 1
        report["splits"][split_name] = {
            "bad_plumes": len(bad),
            "manifest_rows_before": before,
            "manifest_rows_after": after,
            "removed_dirs": removed_dirs,
        }

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
