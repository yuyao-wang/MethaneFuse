import argparse
import csv
import importlib.util
import inspect
import json
import os
import re
import sys
import types
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import torch
import torch.nn.functional as F

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

PRITHVI_REQUIRED_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
ANYSAT_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
ANYSAT_L8_ORDER = ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
ANYSAT_L89_ABLATION = [("B1", 1), ("B2", 2), ("B3", 3), ("B4", 4), ("B5", 5), ("B6", 6), ("B7", 7)]
SATMAE_DEFAULT_MODEL = "vit_base_patch16"
SCALEMAE_DEFAULT_MODEL = "vit_base_patch16"
SCALEMAE_DEFAULT_REPO_URL = "https://github.com/bair-climate-initiative/scale-mae/archive/refs/heads/main.zip"
EARTHPT_DEFAULT_REPO_URL = "https://github.com/aspiaspace/EarthPT/archive/refs/heads/main.zip"
EARTHPT_DEFAULT_CKPT_URL = "https://huggingface.co/Smith42/EarthPT/resolve/main/010M_ckpt.pt"


def canonical_band_id(band_label: str) -> str:
    token = band_label.split("/")[-1].upper()
    if token.startswith("SR_"):
        token = token[3:]
    # Normalize common spellings: B02->B2, B08A->B8A, etc.
    m = re.match(r"^B0*([0-9]+)(A?)$", token)
    if m is not None:
        token = f"B{int(m.group(1))}{m.group(2)}"
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


def validate_anysat_settings(input_size: int, anysat_patch_size: int) -> None:
    if anysat_patch_size % 10 != 0:
        raise ValueError("--anysat-patch-size must be a multiple of 10 (e.g., 50, 100).")

    effective_patch = anysat_patch_size // 10
    num_patches = (input_size // effective_patch) ** 2
    if num_patches > 1024:
        raise ValueError(
            f"Current AnySat settings are too memory-heavy: input_size={input_size}, "
            f"anysat_patch_size={anysat_patch_size} -> about {num_patches} patches. "
            "Increase --anysat-patch-size (recommended: 100) or reduce --input-size."
        )


def _load_module_from_file(module_name: str, module_file: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_satmae_root(base: Path) -> Optional[Path]:
    if (base / "models_vit.py").is_file():
        return base
    # common extracted layout: SatMAE-main/models_vit.py
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir() and (child / "models_vit.py").is_file():
                return child
    # shallow fallback search
    for p in base.rglob("models_vit.py"):
        if len(p.relative_to(base).parts) <= 4:
            return p.parent
    return None


def resolve_satmae_repo(
    repo_dir: str,
    cache_dir: Path,
    repo_url: str,
) -> Path:
    if repo_dir:
        root = _find_satmae_root(Path(repo_dir).expanduser().resolve())
        if root is None:
            raise FileNotFoundError(
                f"Could not locate models_vit.py under --satmae-repo-dir={repo_dir}. "
                "Point to SatMAE root or a parent containing it."
            )
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _find_satmae_root(cache_dir)
    if repo_root is not None:
        return repo_root

    if not repo_url:
        raise ValueError("SatMAE repo was not found locally and --satmae-repo-url is empty.")

    zip_path = maybe_download(repo_url, cache_dir / "satmae_repo.zip")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    repo_root = _find_satmae_root(cache_dir)
    if repo_root is None:
        raise FileNotFoundError(
            f"Downloaded SatMAE repo archive but could not find models_vit.py in {cache_dir}."
        )
    return repo_root


def resolve_satmae_checkpoint(
    checkpoint_path: str,
    checkpoint_url: str,
    cache_dir: Path,
) -> Path:
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"SatMAE checkpoint not found: {ckpt}")
        return ckpt

    if checkpoint_url:
        parsed = urlparse(checkpoint_url)
        fname = Path(parsed.path).name or "satmae_checkpoint.pth"
        return maybe_download(checkpoint_url, cache_dir / fname)

    raise ValueError("SatMAE requires --satmae-checkpoint or --satmae-checkpoint-url.")


def prepare_satmae_input(
    sensor: str,
    x_dict: Dict[str, torch.Tensor],
    source_band_labels: Sequence[str],
    device: torch.device,
    requested_bands_csv: str,
    fill_missing_zero: bool,
) -> Tuple[torch.Tensor, List[str]]:
    sensor = sensor.lower()
    imgs = x_dict["imgs"].to(device)

    source_canonical = [canonical_band_id(lbl) for lbl in source_band_labels]
    band_to_idx: Dict[str, int] = {}
    for idx, band in enumerate(source_canonical):
        if band not in band_to_idx:
            band_to_idx[band] = idx

    if requested_bands_csv.strip():
        requested = [canonical_band_id(x.strip()) for x in requested_bands_csv.split(",") if x.strip()]
    else:
        # Default: use all channels provided by the chosen dataset config.
        requested = list(source_canonical)

    missing = [b for b in requested if b not in band_to_idx]
    if missing and not fill_missing_zero:
        raise ValueError(
            f"SatMAE requested bands not present in {sensor} sample: {missing}. "
            f"Available: {list(band_to_idx.keys())}"
        )

    if fill_missing_zero:
        bsz, _, h, w = imgs.shape
        x = torch.zeros((bsz, len(requested), h, w), dtype=imgs.dtype, device=device)
        for out_idx, band in enumerate(requested):
            if band in band_to_idx:
                x[:, out_idx, :, :] = imgs[:, band_to_idx[band], :, :]
    else:
        select_idx = [band_to_idx[b] for b in requested]
        x = imgs[:, select_idx, :, :]
    return x, requested


def _extract_checkpoint_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "student", "teacher"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        return ckpt_obj
    raise RuntimeError(f"Unsupported checkpoint object type: {type(ckpt_obj)}")


def load_satmae_model(
    repo_dir: str,
    model_name: str,
    checkpoint_file: Path,
    in_chans: int,
    img_size: int,
    device: torch.device,
    global_pool: bool,
):
    # SatMAE uses deprecated numpy aliases (e.g., np.float) in pos_embed.py.
    # Add compatibility aliases at runtime for newer NumPy versions.
    try:
        import numpy as np  # type: ignore
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
    except Exception:
        pass

    repo = Path(repo_dir).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"SatMAE repo directory not found: {repo}")
    model_file = repo / "models_vit.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"SatMAE models_vit.py not found in repo: {model_file}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    satmae_mod = _load_module_from_file("satmae_models_vit_runtime", model_file)
    if model_name not in satmae_mod.__dict__:
        known = sorted([k for k in satmae_mod.__dict__.keys() if k.startswith("vit_")])
        raise ValueError(f"Unknown SatMAE model '{model_name}'. Available constructors: {known}")

    model_ctor = satmae_mod.__dict__[model_name]
    model = model_ctor(num_classes=0, in_chans=in_chans, img_size=img_size, global_pool=global_pool)

    try:
        ckpt_obj = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions may not support `weights_only`.
        ckpt_obj = torch.load(checkpoint_file, map_location="cpu")
    except Exception as exc:
        # PyTorch 2.6+ defaults to weights_only=True and may fail for checkpoints
        # that store non-tensor metadata (e.g., argparse.Namespace).
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            print(
                "[SatMAE] weights_only=True failed; retrying with weights_only=False. "
                "Use only with trusted checkpoints."
            )
            ckpt_obj = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        else:
            raise
    state_dict = _extract_checkpoint_state_dict(ckpt_obj)

    # Remove common training wrappers.
    cleaned_state = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned_state[key] = v

    # Interpolate positional embeddings when needed.
    pos_embed_file = repo / "util" / "pos_embed.py"
    if pos_embed_file.is_file():
        try:
            pos_mod = _load_module_from_file("satmae_pos_embed_runtime", pos_embed_file)
            if hasattr(pos_mod, "interpolate_pos_embed"):
                pos_mod.interpolate_pos_embed(model, cleaned_state)
        except Exception as exc:
            print(f"[SatMAE] Warning: failed to interpolate pos_embed automatically: {exc}")

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    for k, v in cleaned_state.items():
        if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            skipped_shape.append(k)

    msg = model.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(f"[SatMAE] Skipped {len(skipped_shape)} mismatched tensors (e.g., {skipped_shape[:3]}).")
    if msg.missing_keys:
        print(f"[SatMAE] Missing keys after load: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[SatMAE] Unexpected keys after load: {len(msg.unexpected_keys)}")

    model.to(device).eval()
    return model


def extract_satmae_feature(model, x_4d: torch.Tensor) -> torch.Tensor:
    feat = model.forward_features(x_4d)
    if isinstance(feat, (tuple, list)):
        feat = next((x for x in feat if isinstance(x, torch.Tensor)), None)
    if not isinstance(feat, torch.Tensor):
        raise RuntimeError(f"Unexpected SatMAE feature type: {type(feat)}")
    if feat.ndim == 3:
        feat = feat[:, 0, :]
    elif feat.ndim == 1:
        feat = feat.unsqueeze(0)
    elif feat.ndim > 2:
        feat = feat.flatten(start_dim=1)
    return feat


def score_channels_by_ablation_satmae(
    model,
    x_4d: torch.Tensor,
    band_labels: Sequence[str],
) -> List[Tuple[str, float, float]]:
    with torch.no_grad():
        baseline = extract_satmae_feature(model, x_4d)

    raw_scores: List[float] = []
    for channel_idx in range(x_4d.shape[1]):
        x_ab = x_4d.clone()
        x_ab[:, channel_idx, :, :] = 0.0
        with torch.no_grad():
            feat_ab = extract_satmae_feature(model, x_ab)
        delta = torch.norm(baseline - feat_ab, dim=1).mean().item()
        raw_scores.append(delta)

    score_sum = sum(raw_scores)
    norm_scores = [s / score_sum if score_sum > 0 else 0.0 for s in raw_scores]
    return list(zip(list(band_labels), raw_scores, norm_scores))


def _find_scalemae_root(base: Path) -> Optional[Path]:
    if (base / "models_vit.py").is_file() or (base / "models_mae.py").is_file():
        return base
    if (base / "src" / "models_vit.py").is_file() or (base / "src" / "models_mae.py").is_file():
        return base / "src"
    if base.is_dir():
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if (child / "models_vit.py").is_file() or (child / "models_mae.py").is_file():
                return child
            if (child / "src" / "models_vit.py").is_file() or (child / "src" / "models_mae.py").is_file():
                return child / "src"
    for p in base.rglob("models_vit.py"):
        if len(p.relative_to(base).parts) <= 5:
            return p.parent
    for p in base.rglob("models_mae.py"):
        if len(p.relative_to(base).parts) <= 5:
            return p.parent
    return None


def resolve_scalemae_repo(
    repo_dir: str,
    cache_dir: Path,
    repo_url: str,
) -> Path:
    if repo_dir:
        root = _find_scalemae_root(Path(repo_dir).expanduser().resolve())
        if root is None:
            raise FileNotFoundError(
                f"Could not locate scale-mae models_vit.py/models_mae.py under --scalemae-repo-dir={repo_dir}."
            )
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _find_scalemae_root(cache_dir)
    if repo_root is not None:
        return repo_root

    if not repo_url:
        raise ValueError("scale-mae repo was not found locally and --scalemae-repo-url is empty.")

    zip_path = maybe_download(repo_url, cache_dir / "scalemae_repo.zip")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    repo_root = _find_scalemae_root(cache_dir)
    if repo_root is None:
        raise FileNotFoundError(
            f"Downloaded scale-mae archive but could not find model sources in {cache_dir}."
        )
    return repo_root


def resolve_scalemae_checkpoint(
    checkpoint_path: str,
    checkpoint_url: str,
    cache_dir: Path,
) -> Path:
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"scale-mae checkpoint not found: {ckpt}")
        return ckpt

    if checkpoint_url:
        parsed = urlparse(checkpoint_url)
        fname = Path(parsed.path).name or "scalemae_checkpoint.pth"
        return maybe_download(checkpoint_url, cache_dir / fname)

    raise ValueError("scale-mae requires --scalemae-checkpoint or --scalemae-checkpoint-url.")


def _infer_scalemae_in_chans(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in state_dict.items():
        if k.endswith("patch_embed.proj.weight") and isinstance(v, torch.Tensor) and v.ndim == 4:
            return int(v.shape[1])
    return None


def load_scalemae_model(
    repo_dir: str,
    model_name: str,
    checkpoint_file: Path,
    in_chans_override: int,
    img_size: int,
    device: torch.device,
    global_pool: bool,
):
    repo = Path(repo_dir).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"scale-mae repo directory not found: {repo}")

    model_file = repo / "models_vit.py"
    if not model_file.is_file():
        model_file = repo / "models_mae.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"scale-mae model source not found under: {repo}")

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(repo.parent) not in sys.path:
        sys.path.insert(0, str(repo.parent))

    scalemae_mod = _load_module_from_file("scalemae_models_runtime", model_file)

    ckpt_obj = _safe_torch_load(checkpoint_file)
    state_dict = _extract_checkpoint_state_dict(ckpt_obj)

    cleaned_state = {}
    for k, v in state_dict.items():
        key = k
        for prefix in ("module.", "model.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned_state[key] = v

    inferred_in_chans = _infer_scalemae_in_chans(cleaned_state)
    in_chans = int(in_chans_override) if in_chans_override > 0 else int(inferred_in_chans or 3)

    if model_name not in scalemae_mod.__dict__:
        known = sorted([k for k in scalemae_mod.__dict__.keys() if k.startswith("vit_")])
        raise ValueError(f"Unknown scale-mae model '{model_name}'. Available constructors: {known}")

    model_ctor = scalemae_mod.__dict__[model_name]
    sig = inspect.signature(model_ctor)
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = 0
    if "in_chans" in sig.parameters:
        kwargs["in_chans"] = in_chans
    if "img_size" in sig.parameters:
        kwargs["img_size"] = img_size
    if "global_pool" in sig.parameters:
        kwargs["global_pool"] = global_pool
    model = model_ctor(**kwargs)

    pos_embed_file = repo / "util" / "pos_embed.py"
    if pos_embed_file.is_file():
        try:
            pos_mod = _load_module_from_file("scalemae_pos_embed_runtime", pos_embed_file)
            if hasattr(pos_mod, "interpolate_pos_embed"):
                pos_mod.interpolate_pos_embed(model, cleaned_state)
        except Exception as exc:
            print(f"[scale-mae] Warning: failed to interpolate pos_embed automatically: {exc}")

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    for k, v in cleaned_state.items():
        if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            skipped_shape.append(k)

    msg = model.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(f"[scale-mae] Skipped {len(skipped_shape)} mismatched tensors (e.g., {skipped_shape[:3]}).")
    if msg.missing_keys:
        print(f"[scale-mae] Missing keys after load: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[scale-mae] Unexpected keys after load: {len(msg.unexpected_keys)}")

    model.to(device).eval()
    return model, dict(in_chans=in_chans)


def prepare_scalemae_input(
    x_dict: Dict[str, torch.Tensor],
    source_band_labels: Sequence[str],
    device: torch.device,
    requested_bands_csv: str,
    fill_missing_zero: bool,
    model_n_chan: int,
) -> Tuple[torch.Tensor, List[str]]:
    imgs = x_dict["imgs"].to(device)
    source_canonical = [canonical_band_id(lbl) for lbl in source_band_labels]
    band_to_idx: Dict[str, int] = {}
    for idx, band in enumerate(source_canonical):
        if band not in band_to_idx:
            band_to_idx[band] = idx

    if requested_bands_csv.strip():
        requested = [canonical_band_id(x.strip()) for x in requested_bands_csv.split(",") if x.strip()]
    else:
        requested = list(source_canonical)

    missing = [b for b in requested if b not in band_to_idx]
    if missing and not fill_missing_zero:
        raise ValueError(
            f"scale-mae requested bands not present in input: {missing}. "
            f"Available: {list(band_to_idx.keys())}"
        )

    if fill_missing_zero:
        bsz, _, h, w = imgs.shape
        x = torch.zeros((bsz, len(requested), h, w), dtype=imgs.dtype, device=device)
        for out_idx, band in enumerate(requested):
            if band in band_to_idx:
                x[:, out_idx, :, :] = imgs[:, band_to_idx[band], :, :]
    else:
        select_idx = [band_to_idx[b] for b in requested]
        x = imgs[:, select_idx, :, :]

    if model_n_chan > 0 and x.shape[1] < model_n_chan:
        bsz, _, h, w = x.shape
        pad_n = model_n_chan - x.shape[1]
        pad = torch.zeros((bsz, pad_n, h, w), dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
        requested = requested + [f"PAD{i+1}" for i in range(pad_n)]
    elif model_n_chan > 0 and x.shape[1] > model_n_chan:
        x = x[:, :model_n_chan, :, :]
        requested = requested[:model_n_chan]

    return x, requested


def extract_scalemae_feature(model, x_4d: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        out = model.forward_features(x_4d)
    else:
        out = model(x_4d)

    if isinstance(out, dict):
        for key in ("x_norm_clstoken", "cls_token", "features", "x", "logits"):
            if key in out and isinstance(out[key], torch.Tensor):
                out = out[key]
                break
    elif isinstance(out, (list, tuple)):
        out = next((x for x in out if isinstance(x, torch.Tensor)), None)

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected scale-mae output type: {type(out)}")
    if out.ndim == 3:
        out = out[:, 0, :]
    elif out.ndim == 4:
        out = out.mean(dim=(2, 3))
    elif out.ndim == 1:
        out = out.unsqueeze(0)
    elif out.ndim > 2:
        out = out.flatten(start_dim=1)
    return out


def score_channels_by_ablation_scalemae(
    model,
    x_4d: torch.Tensor,
    band_labels: Sequence[str],
) -> List[Tuple[str, float, float]]:
    with torch.no_grad():
        baseline = extract_scalemae_feature(model, x_4d)

    raw_scores: List[float] = []
    for channel_idx in range(x_4d.shape[1]):
        x_ab = x_4d.clone()
        x_ab[:, channel_idx, :, :] = 0.0
        with torch.no_grad():
            feat_ab = extract_scalemae_feature(model, x_ab)
        delta = torch.norm(baseline - feat_ab, dim=1).mean().item()
        raw_scores.append(delta)

    score_sum = sum(raw_scores)
    norm_scores = [s / score_sum if score_sum > 0 else 0.0 for s in raw_scores]
    return list(zip(list(band_labels), raw_scores, norm_scores))


def _find_earthpt_root(base: Path) -> Optional[Path]:
    if (base / "model.py").is_file():
        return base
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir() and (child / "model.py").is_file():
                return child
    for p in base.rglob("model.py"):
        if len(p.relative_to(base).parts) <= 4:
            return p.parent
    return None


def _patch_earthpt_model_source(model_file: Path) -> None:
    """
    Patch known upstream EarthPT bug:
    model.py ties weights using `.weight` on nn.Sequential modules, which raises
    AttributeError in current repo snapshot.
    """
    bad = "self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying"
    replacement = (
        "if hasattr(self.transformer.wte, 'weight') and hasattr(self.lm_head, 'weight'):\n"
        "            self.transformer.wte.weight = self.lm_head.weight"
    )

    txt = model_file.read_text(encoding="utf-8")
    if bad not in txt:
        return
    txt = txt.replace(bad, replacement)
    model_file.write_text(txt, encoding="utf-8")
    print("[EarthPT] Applied runtime patch for Sequential weight-tying.")


def resolve_earthpt_repo(
    repo_dir: str,
    cache_dir: Path,
    repo_url: str,
) -> Path:
    if repo_dir:
        root = _find_earthpt_root(Path(repo_dir).expanduser().resolve())
        if root is None:
            raise FileNotFoundError(
                f"Could not locate EarthPT model.py under --earthpt-repo-dir={repo_dir}."
            )
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _find_earthpt_root(cache_dir)
    if repo_root is not None:
        return repo_root

    if not repo_url:
        raise ValueError("EarthPT repo was not found locally and --earthpt-repo-url is empty.")

    zip_path = maybe_download(repo_url, cache_dir / "earthpt_repo.zip")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    repo_root = _find_earthpt_root(cache_dir)
    if repo_root is None:
        raise FileNotFoundError(
            f"Downloaded EarthPT repo archive but could not find model.py in {cache_dir}."
        )
    return repo_root


def resolve_earthpt_checkpoint(
    checkpoint_path: str,
    checkpoint_url: str,
    cache_dir: Path,
) -> Path:
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"EarthPT checkpoint not found: {ckpt}")
        return ckpt

    if checkpoint_url:
        parsed = urlparse(checkpoint_url)
        fname = Path(parsed.path).name or "earthpt_ckpt.pt"
        return maybe_download(checkpoint_url, cache_dir / fname)

    raise ValueError("EarthPT requires --earthpt-checkpoint or --earthpt-checkpoint-url.")


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            print("[EarthPT] weights_only=True failed; retrying with weights_only=False (trusted checkpoints only).")
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def _extract_earthpt_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "student", "teacher"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError(f"Unsupported EarthPT checkpoint format: {type(ckpt_obj)}")


def _infer_earthpt_arch_kwargs(model_module, state_dict: Dict[str, torch.Tensor], ckpt_obj: object) -> Dict[str, int]:
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("model_args"), dict):
        model_args = ckpt_obj["model_args"]
    else:
        model_args = {}

    cfg_cls = model_module.GPTConfig
    sig = inspect.signature(cfg_cls)
    kwargs: Dict[str, int] = {}
    for name in sig.parameters:
        if name in model_args:
            kwargs[name] = model_args[name]

    if "vocab_size" in sig.parameters and "vocab_size" not in kwargs:
        wte = state_dict.get("transformer.wte.weight")
        if isinstance(wte, torch.Tensor):
            kwargs["vocab_size"] = int(wte.shape[0])
        else:
            kwargs["vocab_size"] = 50304
    if "block_size" in sig.parameters and "block_size" not in kwargs:
        kwargs["block_size"] = int(model_args.get("block_size", 1024))
    if "n_layer" in sig.parameters and "n_layer" not in kwargs:
        for k in state_dict:
            if ".h." in k:
                try:
                    layer_idx = int(k.split(".h.")[1].split(".")[0])
                    kwargs["n_layer"] = max(kwargs.get("n_layer", 0), layer_idx + 1)
                except Exception:
                    pass
        kwargs.setdefault("n_layer", int(model_args.get("n_layer", 12)))
    if "n_head" in sig.parameters and "n_head" not in kwargs:
        kwargs["n_head"] = int(model_args.get("n_head", 12))
    if "n_embd" in sig.parameters and "n_embd" not in kwargs:
        wte = state_dict.get("transformer.wte.weight")
        if isinstance(wte, torch.Tensor):
            kwargs["n_embd"] = int(wte.shape[1])
        else:
            kwargs["n_embd"] = int(model_args.get("n_embd", 768))
    if "n_chan" in sig.parameters and "n_chan" not in kwargs:
        if isinstance(state_dict.get("transformer.wte.0.weight"), torch.Tensor):
            w0 = state_dict["transformer.wte.0.weight"]
            # in_features = patch_size*patch_size*n_chan. Infer n_chan once patch_size is known.
            kwargs["n_chan"] = int(model_args.get("n_chan", w0.shape[1]))
        else:
            kwargs["n_chan"] = int(model_args.get("n_chan", 1))
    if "patch_size" in sig.parameters and "patch_size" not in kwargs:
        if isinstance(state_dict.get("transformer.wte.0.weight"), torch.Tensor):
            w0 = state_dict["transformer.wte.0.weight"]
            n_chan_val = int(kwargs.get("n_chan", model_args.get("n_chan", 1)))
            in_features = int(w0.shape[1])
            patch_sq = max(1, in_features // max(1, n_chan_val))
            patch_size = int(round(patch_sq ** 0.5))
            if patch_size * patch_size * n_chan_val != in_features:
                patch_size = 1
            kwargs["patch_size"] = patch_size
        else:
            kwargs["patch_size"] = int(model_args.get("patch_size", 1))
    return kwargs


def load_earthpt_model(
    repo_dir: str,
    checkpoint_file: Path,
    device: torch.device,
):
    repo = Path(repo_dir).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"EarthPT repo directory not found: {repo}")
    model_file = repo / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"EarthPT model.py not found in repo: {model_file}")
    _patch_earthpt_model_source(model_file)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        model_mod = _load_module_from_file("earthpt_model_runtime", model_file)
    except ModuleNotFoundError as exc:
        missing = str(exc)
        if "einops" in missing:
            raise ModuleNotFoundError(
                "EarthPT dependency missing: einops. Install it in your environment, e.g. `pip install einops`."
            ) from exc
        raise
    ckpt_obj = _safe_torch_load(checkpoint_file)
    state_dict = _extract_earthpt_state_dict(ckpt_obj)

    # If checkpoint lacks linear token projector, it likely expects discrete token IDs.
    if "transformer.wte.0.weight" not in state_dict:
        raise RuntimeError(
            "EarthPT checkpoint appears to expect token IDs (no transformer.wte.0.weight found). "
            "This script only supports continuous-input EarthPT checkpoints."
        )

    if not hasattr(model_mod, "GPT") or not hasattr(model_mod, "GPTConfig"):
        raise RuntimeError(
            "EarthPT loader currently expects model.py to define GPT and GPTConfig. "
            "Please provide compatible repo/checkpoint."
        )

    arch_kwargs = _infer_earthpt_arch_kwargs(model_mod, state_dict, ckpt_obj)
    cfg = model_mod.GPTConfig(**arch_kwargs)
    model = model_mod.GPT(cfg)

    cleaned_state = {}
    for k, v in state_dict.items():
        key = k[len("module.") :] if k.startswith("module.") else k
        cleaned_state[key] = v

    msg = model.load_state_dict(cleaned_state, strict=False)
    if msg.missing_keys:
        print(f"[EarthPT] Missing keys after load: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[EarthPT] Unexpected keys after load: {len(msg.unexpected_keys)}")

    model.to(device).eval()
    return model, dict(
        n_chan=int(getattr(model.config, "n_chan", 1)),
        block_size=int(getattr(model.config, "block_size", 256)),
        patch_size=int(getattr(model.config, "patch_size", 1)),
    )


def prepare_earthpt_input(
    sensor: str,
    x_dict: Dict[str, torch.Tensor],
    source_band_labels: Sequence[str],
    device: torch.device,
    requested_bands_csv: str,
    max_tokens: int,
    model_n_chan: int,
    model_block_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    imgs = x_dict["imgs"].to(device)  # B,C,H,W
    source_canonical = [canonical_band_id(lbl) for lbl in source_band_labels]
    band_to_idx: Dict[str, int] = {}
    for idx, band in enumerate(source_canonical):
        if band not in band_to_idx:
            band_to_idx[band] = idx

    if requested_bands_csv.strip():
        requested = [canonical_band_id(x.strip()) for x in requested_bands_csv.split(",") if x.strip()]
    else:
        if sensor.lower() == "s2":
            requested = ["B2", "B3", "B8", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
        else:
            requested = list(source_canonical)

    missing = [b for b in requested if b not in band_to_idx]
    if missing:
        raise ValueError(
            f"EarthPT requested bands not present in sample: {missing}. "
            f"Available: {list(band_to_idx.keys())}"
        )

    select_idx = [band_to_idx[b] for b in requested]
    x = imgs[:, select_idx, :, :]  # B,C,H,W
    if x.shape[1] < model_n_chan:
        pad = torch.zeros(
            (x.shape[0], model_n_chan - x.shape[1], x.shape[2], x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([x, pad], dim=1)
    elif x.shape[1] > model_n_chan:
        x = x[:, :model_n_chan, :, :]
        requested = requested[:model_n_chan]

    bsz, c, h, w = x.shape
    seq = x.permute(0, 2, 3, 1).reshape(bsz, h * w, c)  # B,T,C
    token_cap = model_block_size if max_tokens <= 0 else min(max_tokens, model_block_size)
    if seq.shape[1] > token_cap:
        seq = seq[:, :token_cap, :]
    return seq, requested


def extract_earthpt_feature(model, x_seq: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_embeddings"):
        out = model.get_embeddings(x_seq)
    elif hasattr(model, "forward_features"):
        out = model.forward_features(x_seq)
    else:
        out = model(x_seq)

    if isinstance(out, dict):
        for key in ("features", "last_hidden_state", "x", "logits"):
            if key in out and isinstance(out[key], torch.Tensor):
                out = out[key]
                break
    elif isinstance(out, (list, tuple)):
        out = next((x for x in out if isinstance(x, torch.Tensor)), None)

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected EarthPT output type: {type(out)}")
    if out.ndim == 3:
        out = out.mean(dim=1)
    elif out.ndim == 1:
        out = out.unsqueeze(0)
    elif out.ndim > 2:
        out = out.flatten(start_dim=1)
    return out


def score_channels_by_ablation_earthpt(
    model,
    x_seq: torch.Tensor,
    band_labels: Sequence[str],
) -> List[Tuple[str, float, float]]:
    with torch.no_grad():
        baseline = extract_earthpt_feature(model, x_seq)

    raw_scores: List[float] = []
    for channel_idx in range(x_seq.shape[-1]):
        x_ab = x_seq.clone()
        x_ab[:, :, channel_idx] = 0.0
        with torch.no_grad():
            feat_ab = extract_earthpt_feature(model, x_ab)
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
    valid = {"panopticon", "prithvi", "anysat", "satmae", "earthpt", "scalemae", "both", "all"}
    for m in models:
        if m not in valid:
            raise ValueError(
                f"Unknown model '{m}'. Use panopticon, prithvi, anysat, satmae, earthpt, scalemae, "
                "both, all, or comma-separated list."
            )
    if "all" in models:
        # Keep "all" on stable built-in defaults.
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
        description=(
            "Compute per-band scores on S2/L89 for PanOpticOn, Prithvi, AnySat, "
            "SatMAE, EarthPT, and/or scale-mae via channel ablation."
        )
    )
    parser.add_argument("--sensor", required=True, choices=["s2", "l89"], help="Which sensor dataset config to use.")
    parser.add_argument("--csv", required=True, help="CSV path used by the dataset loader.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index inside this sensor subset.")
    parser.add_argument(
        "--models",
        default="both",
        help=(
            "panopticon, prithvi, anysat, satmae, earthpt, scalemae, "
            "both(panopticon+prithvi), all(excludes satmae/earthpt/scalemae), or comma-separated list."
        ),
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
    parser.add_argument(
        "--anysat-patch-size",
        type=int,
        default=100,
        help="AnySat patch_size argument (meters, multiple of 10; larger uses less memory).",
    )
    parser.add_argument("--anysat-date-doy", type=int, default=0, help="AnySat day-of-year value for t0.")

    parser.add_argument(
        "--satmae-repo-dir",
        default="",
        help="Optional local path to cloned SatMAE repo (must contain models_vit.py). If empty, auto-download is used.",
    )
    parser.add_argument(
        "--satmae-repo-url",
        default="https://github.com/sustainlab-group/SatMAE/archive/refs/heads/main.zip",
        help="SatMAE repo zip URL used when --satmae-repo-dir is not provided.",
    )
    parser.add_argument(
        "--satmae-model",
        default=SATMAE_DEFAULT_MODEL,
        help="SatMAE model constructor name from models_vit.py (e.g., vit_base_patch16).",
    )
    parser.add_argument(
        "--satmae-checkpoint",
        default="",
        help="Local SatMAE checkpoint path (.pth).",
    )
    parser.add_argument(
        "--satmae-checkpoint-url",
        default="",
        help="Optional SatMAE checkpoint URL to auto-download when local checkpoint is not provided.",
    )
    parser.add_argument(
        "--satmae-cache-dir",
        default="checkpoints/satmae_cache",
        help="Cache directory used when downloading SatMAE checkpoints from URL.",
    )
    parser.add_argument(
        "--satmae-bands",
        default="",
        help="Optional comma-separated band list for SatMAE input (e.g., B2,B3,B4,B8). Default: all dataset bands.",
    )
    parser.add_argument(
        "--satmae-fill-missing-zero",
        action="store_true",
        help="If requested SatMAE bands are missing in input (e.g., L89 vs S2), fill missing channels with zeros.",
    )
    parser.add_argument(
        "--satmae-in-chans",
        type=int,
        default=0,
        help="Override SatMAE in_chans. Default (0) uses number of selected bands.",
    )
    parser.add_argument(
        "--satmae-global-pool",
        action="store_true",
        help="Enable SatMAE global_pool model option (some checkpoints use this).",
    )

    parser.add_argument(
        "--scalemae-repo-dir",
        default="",
        help="Optional local path to scale-mae repo (must contain models_vit.py or models_mae.py). If empty, auto-download is used.",
    )
    parser.add_argument(
        "--scalemae-repo-url",
        default=SCALEMAE_DEFAULT_REPO_URL,
        help="scale-mae repo zip URL used when --scalemae-repo-dir is not provided.",
    )
    parser.add_argument(
        "--scalemae-model",
        default=SCALEMAE_DEFAULT_MODEL,
        help="scale-mae model constructor name (e.g., vit_base_patch16).",
    )
    parser.add_argument(
        "--scalemae-checkpoint",
        default="",
        help="Local scale-mae checkpoint path (.pt/.pth).",
    )
    parser.add_argument(
        "--scalemae-checkpoint-url",
        default="",
        help="Optional scale-mae checkpoint URL to auto-download when local checkpoint is not provided.",
    )
    parser.add_argument(
        "--scalemae-cache-dir",
        default="checkpoints/scalemae_cache",
        help="Cache directory for auto-downloaded scale-mae repo/checkpoint.",
    )
    parser.add_argument(
        "--scalemae-bands",
        default="",
        help="Optional comma-separated scale-mae band list (default: all dataset bands).",
    )
    parser.add_argument(
        "--scalemae-fill-missing-zero",
        action="store_true",
        help="If requested scale-mae bands are missing in input, fill missing channels with zeros.",
    )
    parser.add_argument(
        "--scalemae-in-chans",
        type=int,
        default=0,
        help="Override scale-mae in_chans. Default (0) infers from checkpoint patch_embed.",
    )
    parser.add_argument(
        "--scalemae-global-pool",
        action="store_true",
        help="Enable scale-mae global_pool model option when constructor supports it.",
    )

    parser.add_argument(
        "--earthpt-repo-dir",
        default="",
        help="Optional local path to EarthPT repo (must contain model.py). If empty, auto-download is used.",
    )
    parser.add_argument(
        "--earthpt-repo-url",
        default=EARTHPT_DEFAULT_REPO_URL,
        help="EarthPT repo zip URL used when --earthpt-repo-dir is not provided.",
    )
    parser.add_argument(
        "--earthpt-checkpoint",
        default="",
        help="Local EarthPT checkpoint path (.pt/.pth).",
    )
    parser.add_argument(
        "--earthpt-checkpoint-url",
        default=EARTHPT_DEFAULT_CKPT_URL,
        help="EarthPT checkpoint URL used when --earthpt-checkpoint is not provided.",
    )
    parser.add_argument(
        "--earthpt-cache-dir",
        default="checkpoints/earthpt_cache",
        help="Cache directory for auto-downloaded EarthPT repo/checkpoint.",
    )
    parser.add_argument(
        "--earthpt-bands",
        default="",
        help="Optional comma-separated EarthPT band list (default: all bands from current dataset config).",
    )
    parser.add_argument(
        "--earthpt-max-tokens",
        type=int,
        default=1024,
        help="Max number of EarthPT tokens per sample after reshaping image to sequence (0 means no cap).",
    )

    args = parser.parse_args()
    models = parse_model_list(args.models)
    device = torch.device(args.device)
    if "anysat" in models:
        validate_anysat_settings(args.input_size, args.anysat_patch_size)

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

    if "satmae" in models:
        print("Loading sample for SatMAE (scale_to_unit=True)...")
        x_sat, sat_source_labels, resolved_idx_sat = load_dataset_sample(
            sensor=args.sensor,
            csv_path=args.csv,
            sample_index=args.sample_index,
            scale_to_unit=True,
            target_size=args.input_size,
        )
        x_satmae, satmae_labels = prepare_satmae_input(
            sensor=args.sensor,
            x_dict=x_sat,
            source_band_labels=sat_source_labels,
            device=device,
            requested_bands_csv=args.satmae_bands,
            fill_missing_zero=args.satmae_fill_missing_zero,
        )
        in_chans = args.satmae_in_chans if args.satmae_in_chans > 0 else int(x_satmae.shape[1])
        if in_chans != int(x_satmae.shape[1]):
            raise ValueError(
                f"--satmae-in-chans={in_chans} does not match selected SatMAE bands ({int(x_satmae.shape[1])})."
            )

        satmae_ckpt = resolve_satmae_checkpoint(
            checkpoint_path=args.satmae_checkpoint,
            checkpoint_url=args.satmae_checkpoint_url,
            cache_dir=Path(args.satmae_cache_dir),
        )
        satmae_repo = resolve_satmae_repo(
            repo_dir=args.satmae_repo_dir,
            cache_dir=Path(args.satmae_cache_dir) / "repo_src",
            repo_url=args.satmae_repo_url,
        )
        print(f"Using SatMAE repo: {satmae_repo}")
        print(f"Using SatMAE checkpoint: {satmae_ckpt}")
        print("Loading SatMAE model...")
        satmae_model = load_satmae_model(
            repo_dir=str(satmae_repo),
            model_name=args.satmae_model,
            checkpoint_file=satmae_ckpt,
            in_chans=in_chans,
            img_size=args.input_size,
            device=device,
            global_pool=args.satmae_global_pool,
        )
        satmae_scores = score_channels_by_ablation_satmae(
            satmae_model,
            x_4d=x_satmae,
            band_labels=satmae_labels,
        )
        print_scores("SatMAE", satmae_scores, args.topk)
        if args.out_csv:
            save_scores_csv(Path(args.out_csv), "satmae", args.sensor, resolved_idx_sat, satmae_scores)

    if "scalemae" in models:
        print("Loading sample for scale-mae (scale_to_unit=True)...")
        x_scale, scale_source_labels, resolved_idx_scale = load_dataset_sample(
            sensor=args.sensor,
            csv_path=args.csv,
            sample_index=args.sample_index,
            scale_to_unit=True,
            target_size=args.input_size,
        )
        scalemae_repo = resolve_scalemae_repo(
            repo_dir=args.scalemae_repo_dir,
            cache_dir=Path(args.scalemae_cache_dir) / "repo_src",
            repo_url=args.scalemae_repo_url,
        )
        scalemae_ckpt = resolve_scalemae_checkpoint(
            checkpoint_path=args.scalemae_checkpoint,
            checkpoint_url=args.scalemae_checkpoint_url,
            cache_dir=Path(args.scalemae_cache_dir),
        )
        print(f"Using scale-mae repo: {scalemae_repo}")
        print(f"Using scale-mae checkpoint: {scalemae_ckpt}")
        print("Loading scale-mae model...")
        scalemae_model, scalemae_meta = load_scalemae_model(
            repo_dir=str(scalemae_repo),
            model_name=args.scalemae_model,
            checkpoint_file=scalemae_ckpt,
            in_chans_override=args.scalemae_in_chans,
            img_size=args.input_size,
            device=device,
            global_pool=args.scalemae_global_pool,
        )
        x_scalemae, scalemae_labels = prepare_scalemae_input(
            x_dict=x_scale,
            source_band_labels=scale_source_labels,
            device=device,
            requested_bands_csv=args.scalemae_bands,
            fill_missing_zero=args.scalemae_fill_missing_zero,
            model_n_chan=int(scalemae_meta.get("in_chans", 0)),
        )
        scalemae_scores = score_channels_by_ablation_scalemae(
            scalemae_model,
            x_4d=x_scalemae,
            band_labels=scalemae_labels,
        )
        print_scores("scale-mae", scalemae_scores, args.topk)
        if args.out_csv:
            save_scores_csv(Path(args.out_csv), "scalemae", args.sensor, resolved_idx_scale, scalemae_scores)

    if "earthpt" in models:
        print("Loading sample for EarthPT (scale_to_unit=True)...")
        x_earth, earth_source_labels, resolved_idx_earth = load_dataset_sample(
            sensor=args.sensor,
            csv_path=args.csv,
            sample_index=args.sample_index,
            scale_to_unit=True,
            target_size=args.input_size,
        )

        earth_repo = resolve_earthpt_repo(
            repo_dir=args.earthpt_repo_dir,
            cache_dir=Path(args.earthpt_cache_dir) / "repo_src",
            repo_url=args.earthpt_repo_url,
        )
        earth_ckpt = resolve_earthpt_checkpoint(
            checkpoint_path=args.earthpt_checkpoint,
            checkpoint_url=args.earthpt_checkpoint_url,
            cache_dir=Path(args.earthpt_cache_dir),
        )
        print(f"Using EarthPT repo: {earth_repo}")
        print(f"Using EarthPT checkpoint: {earth_ckpt}")

        print("Loading EarthPT model...")
        earth_model, earth_meta = load_earthpt_model(
            repo_dir=str(earth_repo),
            checkpoint_file=earth_ckpt,
            device=device,
        )
        x_earth_seq, earth_labels = prepare_earthpt_input(
            sensor=args.sensor,
            x_dict=x_earth,
            source_band_labels=earth_source_labels,
            device=device,
            requested_bands_csv=args.earthpt_bands,
            max_tokens=args.earthpt_max_tokens,
            model_n_chan=int(earth_meta["n_chan"]),
            model_block_size=int(earth_meta["block_size"]),
        )
        earth_scores = score_channels_by_ablation_earthpt(
            earth_model,
            x_seq=x_earth_seq,
            band_labels=earth_labels,
        )
        print_scores("EarthPT", earth_scores, args.topk)
        if args.out_csv:
            save_scores_csv(Path(args.out_csv), "earthpt", args.sensor, resolved_idx_earth, earth_scores)


if __name__ == "__main__":
    main()
