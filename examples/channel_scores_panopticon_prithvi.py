import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

PRITHVI_REQUIRED_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
ANYSAT_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
ANYSAT_L8_ORDER = ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
ANYSAT_L89_ABLATION = [("B1", 1), ("B2", 2), ("B3", 3), ("B4", 4), ("B5", 5), ("B6", 6), ("B7", 7)]


def canonical_band_id(band_label: str) -> str:
    token = band_label.split("/")[-1].upper()
    if token.startswith("SR_"):
        token = token[3:]
    token = re.sub(r"^B0([0-9])$", r"B\1", token)
    return token


def infer_prithvi_checkpoint_name(repo_id: str) -> str:
    key = repo_id.lower()
    if "600m-tl" in key:
        return "Prithvi_EO_V2_600M_TL.pt"
    if "600m" in key:
        return "Prithvi_EO_V2_600M.pt"
    if "300m-tl" in key:
        return "Prithvi_EO_V2_300M_TL.pt"
    return "Prithvi_EO_V2_300M.pt"


def maybe_download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    print(f"Downloading: {url}")
    torch.hub.download_url_to_file(url, str(dst), progress=True)
    return dst


def resolve_prithvi_artifacts(
    repo_id: str,
    cache_dir: Path,
    module_path: str,
    config_path: str,
    checkpoint_path: str,
) -> Tuple[Path, Path, Path]:
    if module_path and config_path and checkpoint_path:
        return Path(module_path), Path(config_path), Path(checkpoint_path)

    repo_cache_dir = cache_dir / repo_id.replace("/", "__")
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    module = Path(module_path) if module_path else maybe_download(
        f"{base_url}/prithvi_mae.py", repo_cache_dir / "prithvi_mae.py"
    )
    config = Path(config_path) if config_path else maybe_download(
        f"{base_url}/config.json", repo_cache_dir / "config.json"
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path else maybe_download(
        f"{base_url}/{infer_prithvi_checkpoint_name(repo_id)}",
        repo_cache_dir / infer_prithvi_checkpoint_name(repo_id),
    )

    return module, config, checkpoint


def load_dataset_sample(
    sensor: str,
    csv_path: str,
    sample_index: int,
    scale_to_unit: bool,
    target_size: int,
) -> Tuple[Dict[str, torch.Tensor], List[str], int]:
    from dinov2.utils.data import load_ds_cfg

    sensor = sensor.lower()
    if sensor == "s2":
        from dinov2.data.datasets.s2_csv import S2CsvDataset

        ds_cfg_name = "s2_12band"
        ds = S2CsvDataset(
            csv_path,
            ds_cfg_name=ds_cfg_name,
            normalize_stats=None,
            scale_to_unit=scale_to_unit,
            pad_to_multiple=None,
        )
    elif sensor == "l89":
        from dinov2.data.datasets.landsat_csv import Landsat89CsvDataset

        ds_cfg_name = "landsat89_7band"
        ds = Landsat89CsvDataset(
            csv_path,
            ds_cfg_name=ds_cfg_name,
            normalize_stats=None,
            scale_to_unit=scale_to_unit,
            pad_to_multiple=None,
        )
    else:
        raise ValueError("--sensor must be one of: s2, l89")

    resolved_index = sample_index
    if hasattr(ds, "df") and "sensor" in ds.df.columns:
        sensor_rows = ds.df.index[ds.df["sensor"].astype(str).str.lower() == sensor].tolist()
        if not sensor_rows:
            raise ValueError(f"CSV has a 'sensor' column but no rows for sensor='{sensor}'.")
        if sample_index < 0 or sample_index >= len(sensor_rows):
            raise IndexError(
                f"sample_index={sample_index} is out of range for sensor='{sensor}' with {len(sensor_rows)} rows."
            )
        resolved_index = sensor_rows[sample_index]

    x_dict, _ = ds[resolved_index]
    x_dict = {k: v.unsqueeze(0) for k, v in x_dict.items()}
    x_dict["imgs"] = F.interpolate(
        x_dict["imgs"].float(), size=(target_size, target_size), mode="bilinear", align_corners=False
    )

    band_cfg = load_ds_cfg(ds_cfg_name)
    band_labels = [b["id"] for b in band_cfg["bands"]]
    return x_dict, band_labels, resolved_index


def load_panopticon_model(weights_path: str, device: torch.device):
    from hubconf import _panopticon_vitb14

    model = _panopticon_vitb14()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def extract_panopticon_cls(model, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(x_dict)
    if isinstance(out, dict):
        if "x_norm_clstoken" not in out:
            raise RuntimeError("Panopticon output dict does not contain 'x_norm_clstoken'.")
        return out["x_norm_clstoken"]
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected Panopticon output type: {type(out)}")
    return out


def score_channels_by_ablation_panopticon(
    model,
    x_dict: Dict[str, torch.Tensor],
    band_labels: Sequence[str],
    device: torch.device,
) -> List[Tuple[str, float, float]]:
    x_dev = {k: v.to(device) for k, v in x_dict.items()}
    imgs = x_dev["imgs"]

    with torch.no_grad():
        baseline = extract_panopticon_cls(model, x_dev)

    raw_scores: List[float] = []
    for channel_idx in range(imgs.shape[1]):
        x_ab = {k: v for k, v in x_dev.items()}
        x_ab["imgs"] = imgs.clone()
        x_ab["imgs"][:, channel_idx, :, :] = 0.0
        with torch.no_grad():
            feat_ab = extract_panopticon_cls(model, x_ab)
        delta = torch.norm(baseline - feat_ab, dim=1).mean().item()
        raw_scores.append(delta)

    score_sum = sum(raw_scores)
    norm_scores = [s / score_sum if score_sum > 0 else 0.0 for s in raw_scores]
    return list(zip(band_labels, raw_scores, norm_scores))


def load_prithvi_model(
    module_file: Path,
    config_file: Path,
    checkpoint_file: Path,
    device: torch.device,
):
    for p in (module_file, config_file, checkpoint_file):
        if not p.is_file():
            raise FileNotFoundError(f"Prithvi artifact not found: {p}")

    module_name = "prithvi_mae_runtime"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Prithvi module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except TypeError as exc:
        # Some Python versions cannot evaluate PEP604 annotations such as
        # tuple[int, int, int] | list[int] at import time. Re-exec with
        # postponed evaluation of annotations for compatibility.
        if "unsupported operand type(s) for |" not in str(exc):
            raise
        source = module_file.read_text(encoding="utf-8")
        if "from __future__ import annotations" not in source:
            lines = source.splitlines()
            insert_at = 0
            if lines and lines[0].startswith("#!"):
                insert_at = 1
            if insert_at < len(lines) and lines[insert_at].startswith("#") and "coding" in lines[insert_at]:
                insert_at += 1
            lines.insert(insert_at, "from __future__ import annotations")
            source = "\n".join(lines) + "\n"

        module = types.ModuleType(module_name)
        module.__file__ = str(module_file)
        module.__package__ = ""
        exec(compile(source, str(module_file), "exec"), module.__dict__)

    if not hasattr(module, "PrithviMAE"):
        raise RuntimeError(f"Prithvi module {module_file} does not define PrithviMAE")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)["pretrained_cfg"]

    model = module.PrithviMAE(**cfg)

    try:
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_file, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_file}")

    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, cfg


def prepare_prithvi_input(
    x_dict: Dict[str, torch.Tensor],
    source_band_labels: Sequence[str],
    prithvi_cfg: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    imgs = x_dict["imgs"].to(device)
    band_to_idx = {canonical_band_id(lbl): idx for idx, lbl in enumerate(source_band_labels)}

    missing = [b for b in PRITHVI_REQUIRED_BANDS if b not in band_to_idx]
    if missing:
        raise ValueError(
            f"Input is missing bands required by Prithvi: {missing}. "
            f"Available: {list(band_to_idx.keys())}"
        )

    select_idx = [band_to_idx[b] for b in PRITHVI_REQUIRED_BANDS]
    x = imgs[:, select_idx, :, :]

    mean = torch.tensor(prithvi_cfg["mean"], dtype=x.dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor(prithvi_cfg["std"], dtype=x.dtype, device=device).view(1, -1, 1, 1)
    x = (x - mean) / std

    num_frames = int(prithvi_cfg.get("num_frames", 1))
    x = x.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
    return x, PRITHVI_REQUIRED_BANDS


def extract_prithvi_cls(model, x_5d: torch.Tensor) -> torch.Tensor:
    feats = model.forward_features(x_5d)
    if not isinstance(feats, list) or not feats:
        raise RuntimeError("Prithvi forward_features did not return a non-empty list.")
    last = feats[-1]
    if not isinstance(last, torch.Tensor) or last.ndim != 3:
        raise RuntimeError(f"Unexpected Prithvi feature tensor shape: {getattr(last, 'shape', None)}")
    return last[:, 0, :]


def score_channels_by_ablation_prithvi(
    model,
    x_5d: torch.Tensor,
    band_labels: Sequence[str],
) -> List[Tuple[str, float, float]]:
    with torch.no_grad():
        baseline = extract_prithvi_cls(model, x_5d)

    raw_scores: List[float] = []
    for channel_idx in range(x_5d.shape[1]):
        x_ab = x_5d.clone()
        x_ab[:, channel_idx, :, :, :] = 0.0
        with torch.no_grad():
            feat_ab = extract_prithvi_cls(model, x_ab)
        delta = torch.norm(baseline - feat_ab, dim=1).mean().item()
        raw_scores.append(delta)

    score_sum = sum(raw_scores)
    norm_scores = [s / score_sum if score_sum > 0 else 0.0 for s in raw_scores]
    return list(zip(band_labels, raw_scores, norm_scores))


def load_anysat_model(
    device: torch.device,
    hub_repo: str,
    hub_entry: str,
    flash_attn: bool,
):
    try:
        model = torch.hub.load(
            hub_repo,
            hub_entry,
            pretrained=True,
            flash_attn=flash_attn,
            trust_repo=True,
        )
    except TypeError:
        model = torch.hub.load(
            hub_repo,
            hub_entry,
            pretrained=True,
            flash_attn=flash_attn,
        )
    model.to(device).eval()
    return model


def prepare_anysat_input(
    sensor: str,
    x_dict: Dict[str, torch.Tensor],
    source_band_labels: Sequence[str],
    device: torch.device,
    date_doy: int,
) -> Tuple[Dict[str, torch.Tensor], str, List[str], List[int]]:
    imgs = x_dict["imgs"].to(device)
    batch_size = imgs.shape[0]
    band_to_idx = {canonical_band_id(lbl): idx for idx, lbl in enumerate(source_band_labels)}

    if sensor == "s2":
        missing = [b for b in ANYSAT_S2_BANDS if b not in band_to_idx]
        if missing:
            raise ValueError(
                f"S2 sample is missing bands required by AnySat: {missing}. "
                f"Available: {list(band_to_idx.keys())}"
            )
        select_idx = [band_to_idx[b] for b in ANYSAT_S2_BANDS]
        x = imgs[:, select_idx, :, :].unsqueeze(1)  # B, T=1, C=10, H, W
        dates = torch.full((batch_size, 1), int(date_doy), dtype=torch.long, device=device)
        data = {"s2": x, "s2_dates": dates}
        return data, "s2", list(ANYSAT_S2_BANDS), list(range(len(ANYSAT_S2_BANDS)))

    if sensor == "l89":
        required = [b for b, _ in ANYSAT_L89_ABLATION]
        missing = [b for b in required if b not in band_to_idx]
        if missing:
            raise ValueError(
                f"L89 sample is missing required bands: {missing}. "
                f"Available: {list(band_to_idx.keys())}"
            )
        b, _, h, w = imgs.shape
        l8 = torch.zeros((b, 1, len(ANYSAT_L8_ORDER), h, w), dtype=imgs.dtype, device=device)
        for band, l8_idx in ANYSAT_L89_ABLATION:
            l8[:, 0, l8_idx, :, :] = imgs[:, band_to_idx[band], :, :]
        dates = torch.full((batch_size, 1), int(date_doy), dtype=torch.long, device=device)
        data = {"l8": l8, "l8_dates": dates}
        labels = [f"l89/{band}" for band, _ in ANYSAT_L89_ABLATION]
        ablate_idx = [idx for _, idx in ANYSAT_L89_ABLATION]
        return data, "l8", labels, ablate_idx

    raise ValueError(f"Unsupported sensor for AnySat: {sensor}")


def extract_anysat_feature(model, data: Dict[str, torch.Tensor], patch_size: int) -> torch.Tensor:
    feat = model(data, patch_size=patch_size, output="tile")
    if isinstance(feat, (list, tuple)):
        feat = next((x for x in feat if isinstance(x, torch.Tensor)), None)
    if not isinstance(feat, torch.Tensor):
        raise RuntimeError(f"Unexpected AnySat output type: {type(feat)}")
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
    elif feat.ndim > 2:
        feat = feat.flatten(start_dim=1)
    return feat


def score_channels_by_ablation_anysat(
    model,
    data: Dict[str, torch.Tensor],
    modality_key: str,
    band_labels: Sequence[str],
    ablate_indices: Sequence[int],
    patch_size: int,
) -> List[Tuple[str, float, float]]:
    with torch.no_grad():
        baseline = extract_anysat_feature(model, data, patch_size=patch_size)

    raw_scores: List[float] = []
    for label, channel_idx in zip(band_labels, ablate_indices):
        data_ab = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        data_ab[modality_key][:, :, channel_idx, :, :] = 0.0
        with torch.no_grad():
            feat_ab = extract_anysat_feature(model, data_ab, patch_size=patch_size)
        delta = torch.norm(baseline - feat_ab, dim=1).mean().item()
        raw_scores.append(delta)

    score_sum = sum(raw_scores)
    norm_scores = [s / score_sum if score_sum > 0 else 0.0 for s in raw_scores]
    return list(zip(list(band_labels), raw_scores, norm_scores))


def print_scores(title: str, scores: Sequence[Tuple[str, float, float]], topk: int) -> None:
    print(f"\n[{title}]")
    print("Band            raw_delta      norm_score")
    print("------------------------------------------")
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    if topk > 0:
        ranked = ranked[:topk]
    for band, raw, norm in ranked:
        print(f"{band:15s} {raw:12.6f} {norm:12.6f}")


def save_scores_csv(
    output_file: Path,
    model_name: str,
    sensor: str,
    resolved_index: int,
    scores: Sequence[Tuple[str, float, float]],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists()
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "sensor", "resolved_index", "rank", "band", "raw_delta", "norm_score"])
        for rank, (band, raw, norm) in enumerate(ranked, start=1):
            writer.writerow([model_name, sensor, resolved_index, rank, band, f"{raw:.8f}", f"{norm:.8f}"])


def parse_model_list(text: str) -> List[str]:
    models = [m.strip().lower() for m in text.split(",") if m.strip()]
    valid = {"panopticon", "prithvi", "anysat", "both", "all"}
    for m in models:
        if m not in valid:
            raise ValueError(
                f"Unknown model '{m}'. Use panopticon, prithvi, anysat, both, all, or comma-separated list."
            )
    if "all" in models:
        return ["panopticon", "prithvi", "anysat"]
    if "both" in models:
        return ["panopticon", "prithvi"]
    out = []
    for m in models:
        if m not in out:
            out.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-band scores on S2/L89 for PanOpticOn, Prithvi, and/or AnySat via channel ablation."
    )
    parser.add_argument("--sensor", required=True, choices=["s2", "l89"], help="Which sensor dataset config to use.")
    parser.add_argument("--csv", required=True, help="CSV path used by the dataset loader.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index inside this sensor subset.")
    parser.add_argument(
        "--models",
        default="both",
        help="panopticon, prithvi, anysat, both(panopticon+prithvi), all, or comma-separated list.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Resize H/W to this size before model forward.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--topk", type=int, default=0, help="Print only top-k bands (0 means all).")
    parser.add_argument("--out-csv", default="", help="Optional output CSV file to append scores.")

    parser.add_argument(
        "--panopticon-weights",
        default="weights/panopticon_vitb14_teacher.pth",
        help="Checkpoint for PanOpticOn teacher.",
    )

    parser.add_argument(
        "--prithvi-repo-id",
        default="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        help="Hugging Face repo id for Prithvi.",
    )
    parser.add_argument(
        "--prithvi-cache-dir",
        default="checkpoints/prithvi_cache",
        help="Where auto-downloaded Prithvi files are cached.",
    )
    parser.add_argument("--prithvi-module", default="", help="Optional local path to prithvi_mae.py")
    parser.add_argument("--prithvi-config", default="", help="Optional local path to config.json")
    parser.add_argument("--prithvi-checkpoint", default="", help="Optional local path to .pt checkpoint")

    parser.add_argument("--anysat-hub-repo", default="gastruc/anysat", help="Torch hub repo for AnySat.")
    parser.add_argument("--anysat-hub-entry", default="anysat", help="Hub entrypoint name.")
    parser.add_argument(
        "--anysat-flash-attn",
        action="store_true",
        help="Enable flash attention in AnySat load (default disabled for compatibility).",
    )
    parser.add_argument("--anysat-patch-size", type=int, default=10, help="AnySat patch_size argument.")
    parser.add_argument("--anysat-date-doy", type=int, default=0, help="AnySat day-of-year value for t0.")

    args = parser.parse_args()
    models = parse_model_list(args.models)
    device = torch.device(args.device)

    if "panopticon" in models:
        print("Loading sample for PanOpticOn (scale_to_unit=True)...")
        x_pan, band_labels_pan, resolved_idx_pan = load_dataset_sample(
            sensor=args.sensor,
            csv_path=args.csv,
            sample_index=args.sample_index,
            scale_to_unit=True,
            target_size=args.input_size,
        )

        print("Loading PanOpticOn model...")
        pan_model = load_panopticon_model(args.panopticon_weights, device)
        pan_scores = score_channels_by_ablation_panopticon(
            pan_model,
            x_pan,
            band_labels_pan,
            device,
        )
        print_scores("PanOpticOn", pan_scores, args.topk)
        if args.out_csv:
            save_scores_csv(Path(args.out_csv), "panopticon", args.sensor, resolved_idx_pan, pan_scores)

    if "prithvi" in models:
        print("Loading sample for Prithvi (scale_to_unit=False)...")
        x_raw, source_band_labels, resolved_idx_pri = load_dataset_sample(
            sensor=args.sensor,
            csv_path=args.csv,
            sample_index=args.sample_index,
            scale_to_unit=False,
            target_size=args.input_size,
        )

        module_file, config_file, checkpoint_file = resolve_prithvi_artifacts(
            repo_id=args.prithvi_repo_id,
            cache_dir=Path(args.prithvi_cache_dir),
            module_path=args.prithvi_module,
            config_path=args.prithvi_config,
            checkpoint_path=args.prithvi_checkpoint,
        )
        print(f"Using Prithvi module: {module_file}")
        print(f"Using Prithvi config: {config_file}")
        print(f"Using Prithvi checkpoint: {checkpoint_file}")

        print("Loading Prithvi model...")
        prithvi_model, prithvi_cfg = load_prithvi_model(module_file, config_file, checkpoint_file, device)

        x_prithvi, prithvi_band_labels = prepare_prithvi_input(
            x_dict=x_raw,
            source_band_labels=source_band_labels,
            prithvi_cfg=prithvi_cfg,
            device=device,
        )
        prithvi_scores = score_channels_by_ablation_prithvi(prithvi_model, x_prithvi, prithvi_band_labels)
        print_scores("Prithvi", prithvi_scores, args.topk)
        if args.out_csv:
            save_scores_csv(Path(args.out_csv), "prithvi", args.sensor, resolved_idx_pri, prithvi_scores)

    if "anysat" in models:
        print("Loading sample for AnySat (scale_to_unit=True)...")
        x_any, any_source_labels, resolved_idx_any = load_dataset_sample(
            sensor=args.sensor,
            csv_path=args.csv,
            sample_index=args.sample_index,
            scale_to_unit=True,
            target_size=args.input_size,
        )
        any_data, modality_key, any_labels, any_ablate_idx = prepare_anysat_input(
            sensor=args.sensor,
            x_dict=x_any,
            source_band_labels=any_source_labels,
            device=device,
            date_doy=args.anysat_date_doy,
        )

        print("Loading AnySat model...")
        anysat_model = load_anysat_model(
            device=device,
            hub_repo=args.anysat_hub_repo,
            hub_entry=args.anysat_hub_entry,
            flash_attn=args.anysat_flash_attn,
        )
        anysat_scores = score_channels_by_ablation_anysat(
            anysat_model,
            data=any_data,
            modality_key=modality_key,
            band_labels=any_labels,
            ablate_indices=any_ablate_idx,
            patch_size=args.anysat_patch_size,
        )
        print_scores("AnySat", anysat_scores, args.topk)
        if args.out_csv:
            save_scores_csv(Path(args.out_csv), "anysat", args.sensor, resolved_idx_any, anysat_scores)


if __name__ == "__main__":
    main()
