import argparse
import sys
from pathlib import Path
from typing import Optional, Union

# Make repository importable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov2.data.datasets.s2_csv import S2CsvDataset


def _parse_subset(val: Optional[str]) -> Optional[Union[int, float]]:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return float(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel mean/std for one or more S2 CSVs (concatenated).")
    parser.add_argument("csv", nargs="+", help="Path(s) to CSV file(s).")
    parser.add_argument(
        "--subset",
        type=_parse_subset,
        default=None,
        help="Limit samples for stats (int count or float fraction).",
    )
    parser.add_argument(
        "--ds-cfg-name",
        default="s2_12band",
        help="Dataset config name under dinov2/configs/data for chn_ids.",
    )
    parser.add_argument(
        "--scale-to-unit",
        action="store_true",
        help="Divide uint16 images by 65535 before computing stats.",
    )
    parser.add_argument(
        "--full-spectra",
        action="store_true",
        help="Include spectral stds in chn_ids if available.",
    )
    args = parser.parse_args()

    ds = S2CsvDataset(
        csv_path=args.csv,
        ds_cfg_name=args.ds_cfg_name,
        full_spectra=args.full_spectra,
        scale_to_unit=args.scale_to_unit,
        compute_stats=True,
        compute_stats_subset=args.subset,
        pad_to_multiple=None,
    )

    mean, std = ds.get_normalize_stats()
    print(f"csvs={args.csv}")
    print(f"scale_to_unit={args.scale_to_unit}, subset={args.subset}")
    print("mean =", mean.tolist())
    print("std  =", std.tolist())


if __name__ == "__main__":
    main()
