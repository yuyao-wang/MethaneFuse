import argparse
import csv
import hashlib
import sys
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset

DEFAULT_WV3_BANDS = [
    "Coastal (MS7)",
    "Blue (MS4)",
    "Green (MS3)",
    "Yellow (MS6)",
    "Red (MS2)",
    "Red Edge (MS5)",
    "NIR1 (MS1)",
    "NIR2 (MS8)",
    "SWIR1",
    "SWIR2",
    "SWIR3",
    "SWIR4",
    "SWIR5",
    "SWIR6",
    "SWIR7",
    "SWIR8",
]


def _parse_optional_float(text: str) -> Optional[float]:
    text = str(text).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_wv3_channel_ids_from_srf(
    csv_path: str, band_names: Sequence[str], include_sigmas: bool = False
) -> torch.Tensor:
    """
    Compute WV3 channel IDs (mu or [mu, sigma]) from an SRF CSV file.

    The CSV is expected to have wavelength in the first column and one SRF column per WV3 band.
    """

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"WV3 SRF file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"WV3 SRF file is empty: {csv_path}") from exc

        header_idx = {name: idx for idx, name in enumerate(header) if name}
        missing = [name for name in band_names if name not in header_idx]
        if missing:
            raise ValueError(
                f"Missing WV3 band columns in {csv_path}: {missing}. Available columns include: {header[:18]}"
            )

        wavelengths = []
        responses = {name: [] for name in band_names}
        for row in reader:
            if not row:
                continue
            wavelength = _parse_optional_float(row[0] if len(row) > 0 else "")
            if wavelength is None:
                continue
            wavelengths.append(wavelength)
            for name in band_names:
                idx = header_idx[name]
                value = _parse_optional_float(row[idx] if idx < len(row) else "")
                responses[name].append(max(0.0, value if value is not None else 0.0))

    if len(wavelengths) == 0:
        raise ValueError(f"No numeric wavelength rows parsed from {csv_path}")

    wavelength_t = torch.tensor(wavelengths, dtype=torch.float64)
    mu_list = []
    sigma_list = []
    for name in band_names:
        weights = torch.tensor(responses[name], dtype=torch.float64)
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError(f"Band '{name}' has zero response everywhere in {csv_path}")

        mu = torch.sum(wavelength_t * weights) / total
        mu_list.append(mu)

        if include_sigmas:
            var = torch.sum(((wavelength_t - mu) ** 2) * weights) / total
            sigma_list.append(torch.sqrt(torch.clamp(var, min=0.0)))

    mu_tensor = torch.round(torch.stack(mu_list)).to(torch.int16)
    if not include_sigmas:
        return mu_tensor
    sigma_tensor = torch.round(torch.stack(sigma_list)).to(torch.int16)
    return torch.stack([mu_tensor, sigma_tensor], dim=1)


class StaticAnchoredCache:
    def __init__(self, cache_dir: str, min_free_gb: float = 30.0):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_gb * (1024**3)

    def _get_free_space(self) -> int:
        return shutil.disk_usage(self.cache_dir).free

    def _hashed_path(self, original: str) -> Path:
        norm_path = os.path.abspath(original)
        digest = hashlib.sha1(norm_path.encode("utf-8")).hexdigest()
        subdir = digest[:2]
        suffix = Path(original).suffix
        return self.cache_dir / subdir / f"{digest}{suffix}"

    def ensure_local(self, original: str) -> str:
        dst = self._hashed_path(original)
        if dst.exists():
            return str(dst)
        if self._get_free_space() < self.min_free_bytes:
            return original
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError):
                tmp.unlink()
            return original
        return str(dst)

    def warm_up(self, paths: Sequence[str], max_workers: int = 8) -> None:
        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths:
            return
        print(f"[Cache] Warming up (target: {len(unique_paths)})...", flush=True)

        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            fallback_count = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"[Cache] Warmup complete. Cached: {len(unique_paths) - fallback_count}, Remote: {fallback_count}")


class CLSHead(nn.Module):
    """Minimal DINO-style classifier head: CLS -> LayerNorm -> Linear."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls))


class ConcatTemporalDataset(Dataset):
    """
    Wraps a temporal dataset to concatenate three 16-band timepoints into one 48-band tensor.
    """

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_list, label = self.base_ds[idx]  # list of 3 dicts
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)  # (48, H, W)
        chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)  # (48,) or (48, 2)
        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        return x_dict, label


class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
    """
    S2TemporalCsvDataset wrapper that routes file paths through StaticAnchoredCache.
    Defined at module scope so it is picklable for DataLoader worker processes
    under Python 3.14+ (forkserver/spawn start methods).
    """

    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            path = self._local_file_cache.ensure_local(path)
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)


def load_backbone(weights_path: str, device: torch.device, debug: bool = False):
    """
    Build the Panopticon ViT backbone.

    If ``weights_path`` is "none"/""/None, the model is left randomly initialized
    (i.e., train from scratch). Otherwise the checkpoint at ``weights_path`` is
    loaded. This makes it easy to switch between pretrained and scratch runs via
    ``--weights none`` on the command line.
    """

    from hubconf import _panopticon_vitb14

    if weights_path in (None, "", "none", "scratch", "random"):
        if debug:
            print("Building backbone from scratch (no pretrained weights)", flush=True)
        return _panopticon_vitb14()

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        alt_path = Path(str(weights_path) + "?download=true")
        if alt_path.is_file():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading checkpoint from {weights_path}", flush=True)
    if debug:
        print("Building backbone...", flush=True)
    model = _panopticon_vitb14()
    if debug:
        print("Loading state dict to CPU...", flush=True)
    state = torch.load(weights_path, map_location="cpu")
    if debug:
        print("Applying state dict...", flush=True)
    model.load_state_dict(state, strict=True)
    return model


def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [recursive_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def maybe_wrap_dataparallel(module: nn.Module, device: torch.device, num_gpus: int, module_name: str) -> nn.Module:
    """Wrap ``module`` with DataParallel when multiple GPUs are requested."""

    if device.type != "cuda" or num_gpus <= 1:
        return module

    available = torch.cuda.device_count()
    if available < num_gpus:
        raise RuntimeError(
            f"Requested num_gpus={num_gpus}, but only {available} CUDA device(s) are visible."
        )

    device_ids = list(range(num_gpus))
    print(
        f"Wrapping {module_name} with DataParallel across GPU ids {device_ids}. Batch size will be split automatically.",
        flush=True,
    )
    return nn.DataParallel(module, device_ids=device_ids)


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def resize_imgs_to_224(x_dict):
    """Upsample batch of images in x_dict["imgs"] to 224x224 before the backbone."""

    imgs = x_dict.get("imgs")
    if imgs is None:
        return x_dict

    # Support both (C, H, W) and (B, C, H, W) shapes.
    squeeze_back = False
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
        squeeze_back = True

    if imgs.ndim == 4:
        imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
        if squeeze_back:
            imgs = imgs.squeeze(0)
        x_dict["imgs"] = imgs

    return x_dict


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return False
        return value >= 0.5
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "nan", "none", "null"}:
        return False
    try:
        return float(text) >= 0.5
    except Exception:
        return False


def resolve_overlap_flags_from_df(df, overlap_column: str) -> Tuple[torch.Tensor, str]:
    candidate_columns = []
    if overlap_column:
        candidate_columns.extend([c.strip() for c in overlap_column.split(",") if c.strip()])
    candidate_columns.extend(["overlap_mode", "is_overlap", "overlap", "overlap_flag"])

    seen = set()
    deduped = []
    for col in candidate_columns:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    candidate_columns = deduped

    chosen_col = None
    for col in candidate_columns:
        if col in df.columns:
            chosen_col = col
            break

    if chosen_col is not None:
        flags = torch.tensor([_as_bool(v) for v in df[chosen_col].tolist()], dtype=torch.bool)
        return flags, chosen_col

    count_columns = ["available_sensor_count_single4", "available_sensor_count", "sensor_count", "num_sensors"]
    for col in count_columns:
        if col in df.columns:
            numeric_values = []
            for v in df[col].tolist():
                try:
                    numeric_values.append(float(v))
                except Exception:
                    numeric_values.append(0.0)
            numeric = torch.tensor(numeric_values, dtype=torch.float32)
            flags = numeric >= 2.0
            return flags, f"{col}>=2"

    raise ValueError(
        f"Could not infer overlap flags from test CSV columns. "
        f"Provide --overlap_column and ensure it exists. Available columns: {list(df.columns)}"
    )


def compute_binary_split_metrics(labels: torch.Tensor, preds: torch.Tensor, probs: torch.Tensor) -> dict:
    n = int(labels.numel())
    out = {
        "count": n,
        "acc": float("nan"),
        "fpr": float("nan"),
        "recall": float("nan"),
        "auroc": float("nan"),
    }
    if n == 0:
        return out

    labels = labels.to(torch.int64)
    preds = preds.to(torch.int64)
    out["acc"] = float((preds == labels).float().mean().item())

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    out["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    out["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score

        out["auroc"] = float(roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy()))
    except Exception:
        out["auroc"] = float("nan")
    return out


def compute_eval_split_panel(
    labels: torch.Tensor, preds: torch.Tensor, probs: torch.Tensor, overlap_flags: torch.Tensor
) -> dict:
    if not (labels.shape == preds.shape == probs.shape == overlap_flags.shape):
        raise ValueError("labels/preds/probs/overlap_flags must have identical shape")
    split_masks = {
        "overall": torch.ones_like(labels, dtype=torch.bool),
        "single": ~overlap_flags.to(torch.bool),
        "overlap": overlap_flags.to(torch.bool),
    }
    panel = {}
    for name, mask in split_masks.items():
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        panel[name] = compute_binary_split_metrics(
            labels.index_select(0, idx),
            preds.index_select(0, idx),
            probs.index_select(0, idx),
        )
    return panel


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    run_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": vars(args),
    }
    settings = wandb.Settings(init_timeout=300)

    try:
        return wandb.init(**run_kwargs, settings=settings)
    except Exception as exc:
        print(
            f"[W&B] Online init failed ({type(exc).__name__}: {exc}). Falling back to offline mode.",
            flush=True,
        )

    with suppress(Exception):
        wandb.teardown()

    try:
        run = wandb.init(**run_kwargs, mode="offline", settings=settings)
        print("[W&B] Running in offline mode. Use `wandb sync` to upload later.", flush=True)
        return run
    except Exception as exc:
        print(
            f"[W&B] Offline init failed ({type(exc).__name__}: {exc}). Continuing without W&B logging.",
            flush=True,
        )
        return None


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def default_run_name(args) -> str:
    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    mode = "ft" if args.train_backbone else "head"
    return f"{train_stem}__{test_stem}__{mode}"


def resolve_emit_wv3_path_columns(csv_path: str, requested_cols: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    Resolve temporal path columns for EMIT-simulated WV3 training.

    Priority:
      1) user-requested columns (e.g., path_t0/path_t90/path_t360)
      2) wide-table WV3 columns (wv3_0_path/wv3_90_path/wv3_360_path)
      3) wide-table EMIT columns (emit_0_path/emit_90_path/emit_360_path)
      4) single-frame wide-table fallback (wv3_0_path or emit_0_path, repeated 3x)
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [str(h).strip() for h in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"CSV file has no header row: {csv_path}") from exc

    header_set = set(header)
    if all(col in header_set for col in requested_cols):
        return requested_cols

    wv3_cols = ("wv3_0_path", "wv3_90_path", "wv3_360_path")
    if all(col in header_set for col in wv3_cols):
        print(
            f"[Data] {csv_path}: requested columns {requested_cols} not found; using wide-table WV3 columns {wv3_cols}.",
            flush=True,
        )
        return wv3_cols

    emit_cols = ("emit_0_path", "emit_90_path", "emit_360_path")
    if all(col in header_set for col in emit_cols):
        print(
            f"[Data] {csv_path}: requested columns {requested_cols} not found; using wide-table EMIT columns {emit_cols}.",
            flush=True,
        )
        return emit_cols

    for single_col in ("wv3_0_path", "emit_0_path"):
        if single_col in header_set:
            repeated = (single_col, single_col, single_col)
            print(
                f"[Data] {csv_path}: found only {single_col} in wide-table input; "
                f"using single-frame fallback by repeating it for (t0,t90,t360).",
                flush=True,
            )
            return repeated

    raise ValueError(
        f"Could not resolve temporal EMIT/WV3 path columns for {csv_path}. "
        f"Requested={requested_cols}. Available columns={header}"
    )


def _valid_path_series(path_series):
    text = path_series.astype("string").str.strip().str.lower()
    return (~path_series.isna()) & (~text.isin(["", "nan", "none", "null"]))


def _valid_path_cell(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        try:
            if value != value:  # NaN
                return False
        except Exception:
            return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null"}


def _is_emit_or_wv3_sensor(value) -> bool:
    return str(value).strip().lower() in {"emit", "wv3"}


def build_emit_wv3_prefiltered_csv(
    csv_path: str,
    path_columns: Tuple[str, str, str],
    *,
    split_name: str,
    sensor_column: str = "sensor",
) -> str:
    """
    Build a temporary CSV with rows that have valid EMIT/WV3 path cells.

    This avoids NaN/None path values in wide tables from crashing
    skip-invalid prechecks in the base dataset implementation.
    """

    tmp_dir = os.environ.get("TMPDIR") or None
    fd, tmp_path = tempfile.mkstemp(prefix=f"emit_wv3_prefilter_{split_name}_", suffix=".csv", dir=tmp_dir, text=True)
    os.close(fd)

    unique_path_columns = tuple(dict.fromkeys(path_columns))
    total = 0
    valid_any_sensor = 0
    valid_emit_wv3_sensor = 0

    with open(csv_path, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError(f"{split_name}: CSV has no header: {csv_path}")

        missing = [c for c in unique_path_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{split_name}: missing required path columns {missing} in {csv_path}")

        has_sensor = sensor_column in reader.fieldnames
        for row in reader:
            total += 1
            if not all(_valid_path_cell(row.get(col)) for col in unique_path_columns):
                continue
            valid_any_sensor += 1
            if has_sensor and _is_emit_or_wv3_sensor(row.get(sensor_column)):
                valid_emit_wv3_sensor += 1

    use_sensor_filter = False
    if valid_any_sensor > 0 and valid_emit_wv3_sensor > 0:
        use_sensor_filter = True
    elif valid_any_sensor > 0 and valid_emit_wv3_sensor == 0:
        print(
            f"[Data] {split_name}: no rows matched {sensor_column} in {{emit,wv3}} after path filtering; "
            "ignoring sensor column for wide-table compatibility.",
            flush=True,
        )

    kept = 0
    with open(csv_path, "r", newline="") as fin, open(tmp_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if not all(_valid_path_cell(row.get(col)) for col in unique_path_columns):
                continue
            if use_sensor_filter and not _is_emit_or_wv3_sensor(row.get(sensor_column)):
                continue
            writer.writerow(row)
            kept += 1

    print(
        f"[Data] {split_name}: prefiltered EMIT/WV3 rows {kept}/{total} into temp CSV {tmp_path}.",
        flush=True,
    )
    if kept == 0:
        raise ValueError(f"{split_name}: no valid EMIT/WV3 rows after prefiltering {csv_path}")
    return tmp_path


def filter_dataset_to_emit_wv3_only(base_ds, *, split_name: str, sensor_column: str = "sensor") -> None:
    before = len(base_ds.df)
    df = base_ds.df
    had_sensor_column = sensor_column in df.columns

    if had_sensor_column:
        sensor_vals = df[sensor_column].astype("string").str.strip().str.lower()
        df = df[sensor_vals.isin(["wv3", "emit"])]

    for col in base_ds.path_columns:
        if col not in df.columns:
            raise ValueError(f"{split_name}: missing required path column '{col}' after EMIT/WV3 filtering.")
        df = df[_valid_path_series(df[col])]

    if len(df) == 0 and had_sensor_column:
        fallback_df = base_ds.df
        for col in base_ds.path_columns:
            if col not in fallback_df.columns:
                raise ValueError(f"{split_name}: missing required path column '{col}' during fallback filtering.")
            fallback_df = fallback_df[_valid_path_series(fallback_df[col])]
        if len(fallback_df) > 0:
            print(
                f"[Data] {split_name}: sensor-based filtering produced 0 rows; "
                "falling back to path-only filtering for wide-table compatibility.",
                flush=True,
            )
            df = fallback_df

    base_ds.df = df.reset_index(drop=True)
    after = len(base_ds.df)
    print(
        f"[Data] {split_name}: kept {after}/{before} rows for EMIT/WV3 using path columns {base_ds.path_columns}.",
        flush=True,
    )
    if after == 0:
        raise ValueError(f"{split_name}: no valid EMIT/WV3 rows available after filtering.")


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_test_acc: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "backbone": unwrap_module(backbone).state_dict(),
            "head": unwrap_module(head).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "best_test_acc": best_test_acc,
            "args": vars(args),
        },
        path,
    )


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(
            args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb
        )

    requested_cols = (args.t0_col, args.t90_col, args.t360_col)
    train_path_columns = resolve_emit_wv3_path_columns(args.train_csv, requested_cols)
    test_path_columns = resolve_emit_wv3_path_columns(args.test_csv, requested_cols)

    wv3_band_names = [b.strip() for b in args.wv3_bands.split(",") if b.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one WV3 band column name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(
        args.wv3_srf_csv, wv3_band_names, include_sigmas=args.wv3_full_spectra
    )

    temp_csv_paths = []
    train_csv_for_ds = build_emit_wv3_prefiltered_csv(
        args.train_csv,
        train_path_columns,
        split_name="train",
        sensor_column=args.sensor_column,
    )
    test_csv_for_ds = build_emit_wv3_prefiltered_csv(
        args.test_csv,
        test_path_columns,
        split_name="test",
        sensor_column=args.sensor_column,
    )
    temp_csv_paths.extend([train_csv_for_ds, test_csv_for_ds])

    # Use three WV3 temporal frames as input: t0, t-90, t-360.
    try:
        base_train_ds = CachedS2TemporalCsvDataset(
            csv_path=train_csv_for_ds,
            ds_cfg_name=args.ds_cfg_name,
            normalize_stats=None,  # inputs are already normalized upstream
            scale_to_unit=False,
            pad_to_multiple=args.pad_to_multiple,
            compute_stats=False,
            path_columns=train_path_columns,
            skip_invalid_samples=args.skip_invalid_samples,
            local_file_cache=cache_obj,
        )
        base_test_ds = CachedS2TemporalCsvDataset(
            csv_path=test_csv_for_ds,
            ds_cfg_name=args.ds_cfg_name,
            normalize_stats=None,
            scale_to_unit=False,
            pad_to_multiple=args.pad_to_multiple,
            compute_stats=False,
            path_columns=test_path_columns,
            skip_invalid_samples=args.skip_invalid_samples,
            local_file_cache=cache_obj,
        )
    finally:
        for tmp_csv in temp_csv_paths:
            with suppress(FileNotFoundError):
                os.remove(tmp_csv)

    base_train_ds.chn_ids = wv3_chn_ids
    base_test_ds.chn_ids = wv3_chn_ids.clone()
    filter_dataset_to_emit_wv3_only(base_train_ds, split_name="train", sensor_column=args.sensor_column)
    filter_dataset_to_emit_wv3_only(base_test_ds, split_name="test", sensor_column=args.sensor_column)
    test_overlap_flags, overlap_source = resolve_overlap_flags_from_df(base_test_ds.df, args.overlap_column)
    overlap_count = int(test_overlap_flags.sum().item())
    print(
        f"Loaded overlap flags from '{overlap_source}': overlap={overlap_count}, single={len(test_overlap_flags) - overlap_count}",
        flush=True,
    )
    wv3_mus = base_train_ds.chn_ids[:, 0] if base_train_ds.chn_ids.ndim == 2 else base_train_ds.chn_ids
    print(
        f"WV3 channel IDs loaded from {args.wv3_srf_csv}: bands/timepoint={len(wv3_mus)}, total_temporal_bands={len(wv3_mus) * 3}, "
        f"mu_range=[{int(wv3_mus.min())}, {int(wv3_mus.max())}] nm",
        flush=True,
    )

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for ds in (base_train_ds, base_test_ds):
            for col in ds.path_columns:
                if col in ds.df.columns:
                    all_paths.extend(ds.df[col].dropna().astype(str).tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = ConcatTemporalDataset(base_train_ds)
    test_ds = ConcatTemporalDataset(base_test_ds)
    if len(test_overlap_flags) != len(test_ds):
        raise RuntimeError(
            f"Overlap flags length mismatch: flags={len(test_overlap_flags)} vs test_ds={len(test_ds)}"
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    print(
        f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"train_backbone={args.train_backbone}",
        flush=True,
    )

    backbone = load_backbone(args.weights, device=device, debug=args.debug).to(device)
    head = CLSHead(embed_dim=args.embed_dim, num_classes=2).to(device)

    backbone = maybe_wrap_dataparallel(backbone, device, args.num_gpus, "backbone")

    will_train_backbone = args.train_backbone
    param_groups = [{"params": head.parameters(), "lr": args.head_lr}]
    if will_train_backbone:
        param_groups.insert(0, {"params": backbone.parameters(), "lr": args.backbone_lr})

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    scheduler = build_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    run_name = args.run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_path = ckpt_dir / "ckpt_best_test.pth"
    user_best_ckpt_path = Path(args.best_ckpt_path).expanduser() if args.best_ckpt_path else None
    best_test_acc = float("-inf")
    print(f"Checkpoints will be saved under: {ckpt_dir}", flush=True)
    wandb_run = init_wandb(args)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        freeze_backbone = (not will_train_backbone) or (args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs)
        if freeze_backbone:
            set_trainable(backbone, False)
            backbone.eval()
        else:
            set_trainable(backbone, True)
            backbone.train()
        head.train()

        total_loss = 0.0
        correct = 0
        total = 0
        for step, (x_dict, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)
            # x_dict = resize_imgs_to_224(x_dict)
            

            feats = backbone(x_dict, is_training=True)
            cls_token = feats["x_norm_clstoken"]
            logits = head(cls_token)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if args.log_interval and step % args.log_interval == 0:
                running_loss = total_loss / total
                running_acc = correct / total
                print(
                    f"Epoch {epoch} step {step}/{len(train_loader)} "
                    f"train_loss={running_loss:.4f} train_acc={running_acc:.4f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {"train_loss_running": running_loss, "train_acc_running": running_acc, "step": global_step, "epoch": epoch}
                    )

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        train_loss = total_loss / total
        train_acc = correct / total

        head.eval()
        backbone.eval()
        correct = 0
        total = 0
        test_loss_total = 0.0
        eval_offset = 0
        all_probs = []
        all_preds = []
        all_targets = []
        all_overlap_flags = []
        with torch.no_grad():
            for step, (x_dict, labels) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_dict = recursive_to_device(x_dict, device)
                # x_dict = resize_imgs_to_224(x_dict)
                feats = backbone(x_dict, is_training=True)
                cls_token = feats["x_norm_clstoken"]
                logits = head(cls_token)
                loss = criterion(logits, labels)

                test_loss_total += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_targets.append(labels.detach().cpu())

                batch_size = labels.size(0)
                batch_overlap = test_overlap_flags[eval_offset : eval_offset + batch_size]
                if batch_overlap.numel() != batch_size:
                    raise RuntimeError(
                        f"Overlap flag length mismatch during eval: offset={eval_offset}, "
                        f"batch={batch_size}, flags_total={len(test_overlap_flags)}"
                    )
                all_overlap_flags.append(batch_overlap.clone())
                eval_offset += batch_size

                if args.max_eval_steps is not None and step >= args.max_eval_steps:
                    break

        test_acc = correct / total if total > 0 else float("nan")
        test_loss = test_loss_total / total if total > 0 else float("nan")
        all_probs = torch.cat(all_probs) if len(all_probs) > 0 else torch.tensor([])
        all_preds = torch.cat(all_preds) if len(all_preds) > 0 else torch.tensor([], dtype=torch.int64)
        all_targets = torch.cat(all_targets) if len(all_targets) > 0 else torch.tensor([])
        all_overlap_flags = torch.cat(all_overlap_flags) if len(all_overlap_flags) > 0 else torch.tensor([], dtype=torch.bool)

        if all_targets.numel() > 0:
            split_panel = compute_eval_split_panel(all_targets, all_preds, all_probs, all_overlap_flags)
        else:
            empty = {"count": 0, "acc": float("nan"), "fpr": float("nan"), "recall": float("nan"), "auroc": float("nan")}
            split_panel = {"overall": dict(empty), "single": dict(empty), "overlap": dict(empty)}

        overall_metrics = split_panel["overall"]
        single_metrics = split_panel["single"]
        overlap_metrics = split_panel["overlap"]
        test_acc = overall_metrics["acc"]
        recall = overall_metrics["recall"]
        fpr = overall_metrics["fpr"]
        test_auroc = overall_metrics["auroc"]

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_fpr={fpr:.4f} test_recall={recall:.4f} test_auroc={test_auroc:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][ALL][OVERALL] count={int(overall_metrics['count'])} "
            f"acc={overall_metrics['acc']:.4f} fpr={overall_metrics['fpr']:.4f} "
            f"recall={overall_metrics['recall']:.4f} auroc={overall_metrics['auroc']:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][OVERLAP][OVERALL] count={int(overlap_metrics['count'])} "
            f"acc={overlap_metrics['acc']:.4f} fpr={overlap_metrics['fpr']:.4f} "
            f"recall={overlap_metrics['recall']:.4f} auroc={overlap_metrics['auroc']:.4f}",
            flush=True,
        )
        print(
            f"[EvalSplit][FUSED][SINGLE][OVERALL] count={int(single_metrics['count'])} "
            f"acc={single_metrics['acc']:.4f} fpr={single_metrics['fpr']:.4f} "
            f"recall={single_metrics['recall']:.4f} auroc={single_metrics['auroc']:.4f}",
            flush=True,
        )
        for split_name in ("overall", "single", "overlap"):
            m = split_panel[split_name]
            print(
                f"  test_{split_name}: count={int(m['count'])} acc={m['acc']:.4f} "
                f"fpr={m['fpr']:.4f} recall={m['recall']:.4f} auroc={m['auroc']:.4f}",
                flush=True,
            )

        prev_best_test = best_test_acc
        best_test_acc = max(best_test_acc, test_acc)
        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            backbone,
            head,
            optimizer,
            scheduler,
            best_test_acc,
            args,
        )
        if test_acc > prev_best_test:
            save_checkpoint(
                best_path,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                best_test_acc,
                args,
            )
            print(f"Saved new best checkpoint: {best_path} (test_acc={test_acc:.4f})", flush=True)
            if user_best_ckpt_path is not None:
                save_checkpoint(
                    user_best_ckpt_path,
                    epoch,
                    global_step,
                    backbone,
                    head,
                    optimizer,
                    scheduler,
                    best_test_acc,
                    args,
                )
                print(
                    f"Saved new best checkpoint: {user_best_ckpt_path} (test_acc={test_acc:.4f})",
                    flush=True,
                )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_recall": recall,
                    "test_fpr": fpr,
                    "test_auroc": test_auroc,
                    "test_overall_count": int(overall_metrics["count"]),
                    "test_single_count": int(single_metrics["count"]),
                    "test_single_acc": single_metrics["acc"],
                    "test_single_recall": single_metrics["recall"],
                    "test_single_fpr": single_metrics["fpr"],
                    "test_single_auroc": single_metrics["auroc"],
                    "test_overlap_count": int(overlap_metrics["count"]),
                    "test_overlap_acc": overlap_metrics["acc"],
                    "test_overlap_recall": overlap_metrics["recall"],
                    "test_overlap_fpr": overlap_metrics["fpr"],
                    "test_overlap_auroc": overlap_metrics["auroc"],
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Panopticon ViT + CLS head with concatenated temporal EMIT-simulated WV3 inputs."
    )
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--test_csv", default="")
    parser.add_argument(
        "--weights",
        default="weights/panopticon_vitb14_teacher.pth",
        help='Checkpoint path. Use "none" to train the ViT backbone from scratch.',
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the backbone during finetuning (typically smaller than head lr).",
    )
    parser.add_argument("--lr_scheduler", choices=["none", "noam"], default="noam", help="Learning rate scheduler.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Warmup steps for Noam LR scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (beta1) for Adam.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument(
        "--ds_cfg_name",
        default="landsat89_7band",
        help="Fallback dataset config name (used only to initialize the CSV dataset object).",
    )
    parser.add_argument(
        "--wv3_srf_csv",
        default=str(REPO_ROOT / "WV3_VNIR_SWIR_response.csv"),
        help="Path to WV3 SRF CSV used to derive per-band channel IDs.",
    )
    parser.add_argument(
        "--wv3_bands",
        default=",".join(DEFAULT_WV3_BANDS),
        help="Comma-separated WV3 SRF column names in channel order.",
    )
    parser.add_argument(
        "--wv3_full_spectra",
        action="store_true",
        help="Use full spectra channel IDs [mu, sigma] instead of only mu.",
    )
    parser.add_argument("--device", default="cpu", help='PyTorch device string, e.g. "cpu" or "cuda".')
    parser.add_argument("--debug", action="store_true", help="Print stage timings and device info.")
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N steps.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional cap on number of training batches per epoch (for smoke tests).",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=None,
        help="Optional cap on number of eval batches per epoch (for smoke tests).",
    )
    parser.add_argument(
        "--t0_col",
        default="path_t0",
        help="CSV column for t0 image path. Wide-table fallback supports wv3_0_path / emit_0_path.",
    )
    parser.add_argument(
        "--t90_col",
        default="path_t90",
        help="CSV column for t-90 image path.",
    )
    parser.add_argument(
        "--t360_col",
        default="path_t360",
        help="CSV column for t-360 image path.",
    )
    parser.add_argument(
        "--overlap_column",
        default="overlap_mode",
        help=(
            "CSV column name (or comma-separated candidate names) indicating overlap rows. "
            "Truthy values are treated as overlap; others as single."
        ),
    )
    parser.add_argument(
        "--skip_invalid_samples",
        action="store_true",
        help="Drop CSV rows whose TIFFs are missing or unreadable instead of failing mid-epoch.",
    )
    parser.add_argument(
        "--sensor_column",
        default="sensor",
        help="Optional CSV column with sensor name. If filtering by this column removes all rows, path-only filtering is used.",
    )
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=8)
    parser.add_argument(
        "--local_cache_min_free_gb",
        type=float,
        default=30.0,
        help="Stop caching new files when free disk space goes below this threshold (GB).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="If set, train the entire ViT; otherwise only the classifier head is trained.",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Number of initial epochs to freeze the backbone (only used when train_backbone is True).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm; set <=0 to disable.",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=768,
        help="Backbone output dimension (Panopticon teacher is 768).",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use via torch.nn.DataParallel when --device is CUDA.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Base directory to store checkpoints (latest and best).",
    )
    parser.add_argument(
        "--best_ckpt_path",
        default=None,
        help="Optional explicit file path to save the best-test checkpoint.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run name for checkpoint subfolder. Defaults to <train_csv_stem>__<test_csv_stem>__ft|head.",
    )
    args = parser.parse_args()
    if isinstance(args.best_ckpt_path, str):
        args.best_ckpt_path = args.best_ckpt_path.strip()
        if args.best_ckpt_path == "":
            args.best_ckpt_path = None
    main(args)
