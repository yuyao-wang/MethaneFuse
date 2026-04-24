#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SENSORS: Sequence[str] = ("s2", "l89", "emit", "s5p")
NULL_LIKE = {"", "none", "nan", "null"}


def _clean_path(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in NULL_LIKE:
        return None
    return s


def _load_mask(path: str):
    errors: List[str] = []

    try:
        import tifffile  # type: ignore

        arr = tifffile.imread(path)
        return arr
    except Exception as e:  # pragma: no cover
        errors.append(f"tifffile: {e}")

    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as ds:
            arr = ds.read()
        return arr
    except Exception as e:  # pragma: no cover
        errors.append(f"rasterio: {e}")

    raise RuntimeError(" | ".join(errors))


def _to_2d(arr):
    import numpy as np

    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim < 2:
        raise ValueError(f"mask ndim={a.ndim}, expected >=2")
    axis = int(np.argmin(a.shape))
    return np.take(a, indices=0, axis=axis)


def _collect_rows(
    csv_path: str,
    split: str,
    min_sensors: int,
    require_exists: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor_paths: Dict[str, str] = {}
            for sensor in SENSORS:
                p = _clean_path(row.get(f"{sensor}_plume_path"))
                if p is None:
                    continue
                if require_exists and (not os.path.exists(p)):
                    continue
                sensor_paths[sensor] = p
            if len(sensor_paths) < min_sensors:
                continue

            rows.append(
                {
                    "split": split,
                    "id": str(row.get("id", "")),
                    "plume_id": str(row.get("plume_id", "")),
                    "label": str(row.get("label", "")),
                    "anchor_sensor": str(row.get("anchor_sensor", "")),
                    "sensor_paths": sensor_paths,
                }
            )
    return rows


def _require_dependencies() -> None:
    missing = []

    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")

    try:
        import matplotlib  # noqa: F401
    except Exception:
        missing.append("matplotlib")

    has_reader = False
    try:
        import tifffile  # noqa: F401

        has_reader = True
    except Exception:
        pass
    if not has_reader:
        try:
            import rasterio  # noqa: F401

            has_reader = True
        except Exception:
            pass

    if missing or (not has_reader):
        msg = []
        if missing:
            msg.append("missing: " + ", ".join(missing))
        if not has_reader:
            msg.append("need at least one TIFF reader: tifffile or rasterio")
        raise SystemExit("Dependency check failed: " + "; ".join(msg))


def _draw_one_sample(sample: Dict[str, object], out_png: Path, dpi: int = 150) -> Dict[str, object]:
    import matplotlib.pyplot as plt
    import numpy as np

    split = str(sample["split"])
    sid = str(sample["id"])
    plume_id = str(sample["plume_id"])
    sensor_paths = dict(sample["sensor_paths"])  # type: ignore[arg-type]
    sensors = sorted(sensor_paths.keys(), key=lambda x: list(SENSORS).index(x))

    n = len(sensors)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axes = axes[0]

    stats: Dict[str, Dict[str, object]] = {}
    for ax, sensor in zip(axes, sensors):
        p = str(sensor_paths[sensor])
        arr = _to_2d(_load_mask(p))
        arr = np.asarray(arr)

        pos = int(np.count_nonzero(arr > 0))
        stats[sensor] = {
            "path": p,
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "positive_pixels": pos,
        }

        ax.imshow(arr, cmap="gray", interpolation="nearest")
        ax.set_title(f"{sensor}  pos={pos}", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{split} | id={sid} | plume_id={plume_id} | label={sample['label']} | anchor={sample['anchor_sensor']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    out = dict(sample)
    out["figure"] = str(out_png)
    out["stats"] = stats
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample rows with multi-sensor plume masks and visualize each sensor mask."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--num_samples", type=int, default=8, help="How many rows to sample from combined train+test.")
    parser.add_argument("--min_sensors", type=int, default=2, help="Minimum number of available sensor plume masks.")
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--out_dir", default="outputs/multisensor_mask_viz")
    parser.add_argument(
        "--allow_missing_file",
        action="store_true",
        help="If set, sensor path only needs to be non-empty in CSV; otherwise file must exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_dependencies()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    all_rows.extend(
        _collect_rows(
            csv_path=args.train_csv,
            split="train",
            min_sensors=args.min_sensors,
            require_exists=not args.allow_missing_file,
        )
    )
    all_rows.extend(
        _collect_rows(
            csv_path=args.test_csv,
            split="test",
            min_sensors=args.min_sensors,
            require_exists=not args.allow_missing_file,
        )
    )

    if not all_rows:
        raise SystemExit("No candidate rows found. Check CSV paths, min_sensors, or --allow_missing_file.")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(all_rows))
    picked = rng.sample(all_rows, k=k)

    manifest: List[Dict[str, object]] = []
    for i, sample in enumerate(picked, start=1):
        fig_name = f"{i:03d}_{sample['split']}_id{sample['id']}.png"
        out_png = out_dir / fig_name
        rendered = _draw_one_sample(sample, out_png=out_png)
        manifest.append(rendered)
        print(f"[{i}/{k}] saved: {out_png}")

    summary = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "num_candidates": len(all_rows),
        "num_sampled": k,
        "min_sensors": args.min_sensors,
        "seed": args.seed,
        "out_dir": str(out_dir),
        "samples": manifest,
    }
    manifest_path = out_dir / "sample_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
