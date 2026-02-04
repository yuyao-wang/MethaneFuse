import argparse
import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

try:
    from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample, load_ds_cfg
    from dinov2.utils.data import extract_wavemus
except ImportError:
    class _SkipSample(Exception): pass
    S2TemporalCsvDataset = object
    def extract_wavemus(*args, **kwargs):
        raise ImportError("dinov2 is required for extract_wavemus")


# Landsat 8/9 SR bands (B1-B7) mean/std computed on training split.
L89_PRECOMPUTED_STATS = (
    [10729.92784546, 11384.64407242, 13172.77519667, 14892.25620267, 18149.92169893, 20249.17615773, 18375.0669698],
    [1029.18232283, 1188.52313418, 1552.27685613, 1959.74400972, 1954.80410093, 2098.98682671, 1895.56781996],
)

# Sentinel-2 12 bands mean/std
S2_PRECOMPUTED_STATS = (
    [786.128173828125, 1025.8876953125, 1593.730712890625, 2315.26123046875, 2710.462890625, 3115.90087890625, 3289.0830078125, 3465.536376953125, 3495.579833984375, 3517.7958984375, 4180.28564453125, 3567.866943359375],
    [435.72607421875, 597.6113891601562, 688.5059814453125, 840.1614990234375, 801.7208251953125, 706.9466552734375, 689.823974609375, 727.5567626953125, 668.30224609375, 551.3565063476562, 629.679931640625, 641.590087890625],
)


class StaticAnchoredCache:
    def __init__(self, cache_dir: str, min_free_gb: float = 10.0):
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
        if dst.exists(): return str(dst)
        if self._get_free_space() < self.min_free_bytes: return original
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError): tmp.unlink()
            return original
        return str(dst)

    def warm_up(self, paths: Sequence[str], max_workers: int = 8) -> None:
        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths: return
        print(f"[Cache] Warming up (target: {len(unique_paths)})...", flush=True)
        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            fallback_count = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"[Cache] Warmup complete. Cached: {len(unique_paths)-fallback_count}, Remote: {fallback_count}")

class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            path = self._local_file_cache.ensure_local(path)
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)

class MixedSensorTemporalCsvDataset(S2TemporalCsvDataset):
    """
    A temporal dataset that can handle mixed sensor types (L8/9 and S2)
    by reading a 'sensor' column from the CSV.
    """
    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        # Pop sensor-specific configs that are now handled per-sample
        kwargs.pop("ds_cfg_name", None)
        kwargs.pop("normalize_stats", None)
        self.ds_cfg = None
        self.normalize_stats = None
        self._local_file_cache = local_file_cache

        super().__init__(*args, **kwargs)

        # if "sensor" not in self.df.columns:
            # raise ValueError("CSV must contain a 'sensor' column with values like 'l89' or 's2'.")

        self.sensor_configs = {
            "l89": {
                "ds_cfg": load_ds_cfg("landsat89_7band"),
                "normalize_stats": L89_PRECOMPUTED_STATS,
            },
            "s2": {
                "ds_cfg": load_ds_cfg("s2_12band"),
                "normalize_stats": S2_PRECOMPUTED_STATS,
            },
        }
        # Ensure chn_ids are 2D, i.e. (C, 1)
        for sensor_cfg in self.sensor_configs.values():
            ds_cfg_obj = sensor_cfg["ds_cfg"]
            # Create the chn_ids tensor from the raw config
            chn_ids_tensor = extract_wavemus(ds_cfg_obj, return_sigmas=False)
            chn_ids_tensor = chn_ids_tensor.unsqueeze(-1)  # enforce 2D (C, 1)
            ds_cfg_obj["chn_ids"] = chn_ids_tensor
            sensor_cfg["chn_ids"] = chn_ids_tensor
        # Align channel-id ordering to Sentinel-2 (12 bands); L89 will be padded to match S2 order.
        self._s2_chn_ids = self.sensor_configs["s2"]["chn_ids"]
        self.sensor_configs["l89"]["chn_ids"] = self._s2_chn_ids
        # If inputs are scaled to [0,1], scale the precomputed mean/std once here to avoid double scaling.
        if getattr(self, "scale_to_unit", False):
            for sensor_cfg in self.sensor_configs.values():
                stats = sensor_cfg.get("normalize_stats")
                if stats is None:
                    continue
                mean, std = stats
                sensor_cfg["normalize_stats"] = ([m / 65535.0 for m in mean], [s / 65535.0 for s in std])

    def _load_image(self, path: str, *, column_name: str, sample_id: int):
        """Override to apply sensor-specific normalization and channel IDs."""
        row = self.df.iloc[sample_id]
        sensor = row.get("sensor")
        # sensor = 's2'
        if sensor not in self.sensor_configs:
            raise ValueError(f"Sample {sample_id} has unknown sensor '{sensor}'")

        config = self.sensor_configs[sensor]
        ds_cfg = config["ds_cfg"]
        normalize_stats = config["normalize_stats"]

        # Temporarily set instance-level properties for the base _load_image method
        # This is a bit of a hack, but avoids re-implementing the whole loading logic.
        original_ds_cfg = self.ds_cfg
        original_normalize_stats = self.normalize_stats
        original_local_file_cache = getattr(self, "_local_file_cache", None)

        self.ds_cfg = ds_cfg
        self.normalize_stats = normalize_stats

        try:
            local_cache = getattr(self, "_local_file_cache", None)
            if local_cache is not None and isinstance(path, str):
                path = local_cache.ensure_local(path)
            # The base class's _load_image uses self.ds_cfg and self.normalize_stats
            x_dict = super(MixedSensorTemporalCsvDataset, self)._load_image(
                path, column_name=column_name, sample_id=sample_id
            )
        finally:
            # Restore original properties
            self.ds_cfg = original_ds_cfg
            self.normalize_stats = original_normalize_stats
            if original_local_file_cache is not None:
                self._local_file_cache = original_local_file_cache

        return x_dict

    def __getitem__(self, idx):
        attempts = 0
        last_exc: Optional[Exception] = None

        while attempts < self.max_retries:
            try:
                row = self.df.iloc[idx]
                label = int(row[self.label_column])
                sensor = row.get("sensor")
                # sensor='s2'
                if sensor not in self.sensor_configs:
                    raise ValueError(f"Sample {idx} has unknown sensor '{sensor}'")

                chn_ids = self.sensor_configs[sensor]["chn_ids"]
                sample_id = idx  # use dataframe index for deterministic lookup in _load_image

                x_list = []
                for col in self.path_columns:
                    path = row[col]
                    img = self._load_image(path, column_name=col, sample_id=sample_id)
                    if sensor == "l89":
                        img = self._pad_l89_to_s2(img)
                    # Replace NaN/Inf that can corrupt training
                    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

                    x_dict = dict(imgs=img, chn_ids=chn_ids)
                    if self.transform_each is not None:
                        x_dict = self.transform_each(x_dict)
                    x_list.append(x_dict)

                return x_list, label, sensor

            except _SkipSample as exc:
                last_exc = exc
                attempts += 1
                if attempts == 1:
                    warnings.warn(
                        f"Skipping corrupt temporal sample at row {idx}: {exc}. Retrying with a different index."
                    )
                if attempts >= self.max_retries:
                    raise RuntimeError(f"Exceeded {self.max_retries} retries for temporal index {idx}") from exc
                idx = np.random.randint(0, len(self.df))

        raise RuntimeError("Unreachable: retry loop exited unexpectedly") from last_exc

    @staticmethod
    def _pad_l89_to_s2(img: torch.Tensor) -> torch.Tensor:
        """
        Map 7-channel Landsat 8/9 tensor to Sentinel-2's 12-channel ordering.
        Missing bands are zero-filled; L89 Band 5 goes into S2 Band 8 slot.
        S2 order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
        L89 order: B01,B02,B03,B04,B05,B06,B07
        """
        if img.shape[0] != 7:
            return img
        device, dtype, h, w = img.device, img.dtype, img.shape[1], img.shape[2]
        out = torch.zeros((12, h, w), device=device, dtype=dtype)
        mapping = {
            0: 0,   # coastal
            1: 1,   # blue
            2: 2,   # green
            3: 3,   # red
            4: 7,   # nir -> S2 B08
            5: 10,  # swir1 -> S2 B11
            6: 11,  # swir2 -> S2 B12
        }
        for l89_idx, s2_idx in mapping.items():
            out[s2_idx] = img[l89_idx]
        return out

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
    Wraps the temporal dataset to concatenate three timepoints into one tensor.
    """

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_list, label, sensor = self.base_ds[idx]  # list of 3 dicts, label, and sensor string
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
        chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        return x_dict, label, sensor

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable channel sizes in a batch.
    It pads the 'imgs' spatial dims and channel dims so tensors can be stacked.
    """
    x_dicts, labels, sensors = zip(*batch)

    # Find maxima
    max_channels = max(x['imgs'].shape[0] for x in x_dicts)
    max_h = max(x['imgs'].shape[1] for x in x_dicts)
    max_w = max(x['imgs'].shape[2] for x in x_dicts)

    padded_imgs = []
    padded_chn_ids = []

    for x_dict in x_dicts:
        img = x_dict['imgs']
        chn_ids = x_dict['chn_ids']
        c, h, w = img.shape

        # Pad spatial dims to max_h x max_w
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h < 0 or pad_w < 0:
            raise RuntimeError("Negative padding encountered; check image sizes.")
        img = F.pad(img, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        # Pad channels to max_channels
        pad_c = max_channels - c
        if pad_c < 0:
            raise RuntimeError("Negative channel padding encountered.")
        if pad_c:
            img = torch.cat([img, torch.zeros((pad_c, *img.shape[1:]), device=img.device, dtype=img.dtype)], dim=0)
            chn_ids = torch.cat([chn_ids, torch.zeros((pad_c, *chn_ids.shape[1:]), device=chn_ids.device, dtype=chn_ids.dtype)], dim=0)
        padded_imgs.append(img)

        if chn_ids.ndim != 2:
            raise RuntimeError(f"Unexpected chn_ids dimension: {chn_ids.ndim}. Expected 2D tensor.")
        padded_chn_ids.append(chn_ids)

    batched_x_dict = {'imgs': torch.stack(padded_imgs), 'chn_ids': torch.stack(padded_chn_ids)}
    return batched_x_dict, torch.tensor(labels), list(sensors)

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
    # Accept both pure state_dict and full training checkpoints
    if isinstance(state, dict) and "backbone" in state:
        state = state["backbone"]
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

def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    return run


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def default_run_name(args) -> str:
    """Build a repeatable identifier from datasets and whether backbone trains."""

    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    mode = "ft" if args.train_backbone else "head"
    return f"mixed_{train_stem}__{test_stem}__{mode}"


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_train_acc: float,
    best_test_acc: float,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "best_train_acc": best_train_acc,
            "best_test_acc": best_test_acc,
            "args": vars(args),
        },
        path,
    )


def try_resume(path: Path, backbone: nn.Module, head: nn.Module, optimizer, scheduler, scaler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0

    ckpt = torch.load(path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_train_acc = ckpt.get("best_train_acc", 0.0)
    best_test_acc = ckpt.get("best_test_acc", 0.0)
    print(f"Resumed from {path} at epoch {start_epoch-1}", flush=True)
    return start_epoch, global_step, best_train_acc, best_test_acc


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = device.type == "cuda"
    
    ds_kwargs = {
        # Keep inputs safely within fp16 range (avoid NaNs) by scaling uint16 data to [0,1].
        "scale_to_unit": True,
        "pad_to_multiple": args.pad_to_multiple,
        "compute_stats": False,
        "path_columns": (args.t0_col, args.t90_col, args.t360_col),
        "skip_invalid_samples": args.skip_invalid_samples,
    }

    dataset_cls = MixedSensorTemporalCsvDataset
    cache_obj = None
    if args.local_cache_dir:
        # We can't easily subclass both, so we'll just use the mixed one for now.
        # Caching logic is simple, so it could be merged into MixedSensorTemporalCsvDataset if needed.
        cache_obj = StaticAnchoredCache(args.local_cache_dir)
        ds_kwargs["local_file_cache"] = cache_obj

    base_train_ds = dataset_cls(
        csv_path=args.train_csv,
        **ds_kwargs
    )

    base_test_ds = dataset_cls(
        csv_path=args.test_csv,
        **ds_kwargs
    )

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for col in base_train_ds.path_columns:
            all_paths.extend(base_train_ds.df[col].tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = ConcatTemporalDataset(base_train_ds)
    test_ds = ConcatTemporalDataset(base_test_ds)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn
    )

    print(
        f"Using device={device}, mixed-sensor training, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"train_backbone={args.train_backbone}",
        flush=True,
    )

    backbone = load_backbone(args.weights, device=device, debug=args.debug).to(device)
    head = CLSHead(embed_dim=args.embed_dim, num_classes=2).to(device)
    scaler = GradScaler(enabled=use_amp)

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
    # --- checkpoint bookkeeping ---
    run_name = args.run_name or default_run_name(args)
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    latest_path = ckpt_dir / "ckpt_latest.pth"
    best_train_path = ckpt_dir / "ckpt_best_train.pth"
    best_test_path1 = ckpt_dir / "ckpt_best_test1.pth"
    best_test_path2 = ckpt_dir / "ckpt_best_test2.pth"

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc1 = float("-inf")
    best_test_acc2 = float("-inf")
    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc = try_resume(
            latest_path, backbone, head, optimizer, scheduler, scaler, device
        )
        # Note: best_test_acc1/2 are not restored on resume; they will be recomputed in the new run.
        best_test_acc1 = best_test_acc

    wandb_run = init_wandb(args)

    for epoch in range(start_epoch, args.epochs + 1):
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
        for step, (x_dict, labels, _) in enumerate(train_loader, 1): # Ignore sensor during training
            labels = labels.to(device)
            x_dict = recursive_to_device(x_dict, device)

            with autocast(enabled=use_amp):
                feats = backbone(x_dict, is_training=True)
                cls_token = feats["x_norm_clstoken"]
                logits = head(cls_token)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
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
        sensor_correct = {"l89": 0, "s2": 0}
        sensor_total = {"l89": 0, "s2": 0}
        test_loss_total = 0.0
        tp = fp = fn = tn = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for step, (x_dict, labels, sensors) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_dict = recursive_to_device(x_dict, device)
                with autocast(enabled=use_amp):
                    feats = backbone(x_dict, is_training=True)
                    cls_token = feats["x_norm_clstoken"]
                    logits = head(cls_token)
                    loss = criterion(logits, labels)

                test_loss_total += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                
                for i in range(len(sensors)):
                    sensor_type = sensors[i]
                    sensor_total[sensor_type] += 1
                    if preds[i] == labels[i]:
                        sensor_correct[sensor_type] += 1
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.detach().cpu())
                all_targets.append(labels.detach().cpu())
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

                if args.max_eval_steps is not None and step >= args.max_eval_steps:
                    break

        total = sum(sensor_total.values())
        correct = sum(sensor_correct.values())
        test_acc = correct / total
        test_acc_l89 = sensor_correct["l89"] / sensor_total["l89"] if sensor_total["l89"] > 0 else float("nan")
        test_acc_s2 = sensor_correct["s2"] / sensor_total["s2"] if sensor_total["s2"] > 0 else float("nan")

        test_loss = test_loss_total / total if total > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        all_probs = torch.cat(all_probs) if len(all_probs) > 0 else torch.tensor([])
        all_targets = torch.cat(all_targets) if len(all_targets) > 0 else torch.tensor([])
        if all_probs.numel() > 0:
            try:
                from sklearn.metrics import roc_auc_score

                test_auroc = float(roc_auc_score(all_targets.numpy(), all_probs.numpy()))
            except Exception:
                test_auroc = float("nan")
        else:
            test_auroc = float("nan")

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_acc_l89={test_acc_l89:.4f} test_acc_s2={test_acc_s2:.4f} "
            f"recall={recall:.4f} fpr={fpr:.4f} auroc={test_auroc:.4f} ",
            flush=True,
        )

        # --- save checkpoints ---
        prev_best_train = best_train_acc
        prev_best_test1 = best_test_acc1
        prev_best_test2 = best_test_acc2
        best_train_acc = max(best_train_acc, train_acc)
        if test_acc > best_test_acc1:
            best_test_acc2 = best_test_acc1
            best_test_acc1 = test_acc
            # rotate checkpoints: new best_test_acc1 -> best_test_path1; old best1 -> best2
            save_checkpoint(
                best_test_path2,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                scaler,
                best_train_acc,
                best_test_acc2,
                args,
            )
            save_checkpoint(
                best_test_path1,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                scaler,
                best_train_acc,
                best_test_acc1,
                args,
            )
        elif test_acc > best_test_acc2:
            best_test_acc2 = test_acc
            save_checkpoint(
                best_test_path2,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                scaler,
                best_train_acc,
                best_test_acc2,
                args,
            )

        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            backbone,
            head,
            optimizer,
            scheduler,
            scaler,
            best_train_acc,
            best_test_acc,
            args,
        )

        if train_acc > prev_best_train:
            save_checkpoint(
                best_train_path,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
                scaler,
                best_train_acc,
                best_test_acc1,
                args,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_acc_l89": test_acc_l89,
                    "test_acc_s2": test_acc_s2,
                    "test_recall": recall,
                    "test_fpr": fpr,
                    "test_auroc": test_auroc,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panopticon ViT + CLS head with concatenated temporal inputs from L8/9 or S2.")
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--test_csv", default="", help="CSV for testing. Must also contain a 'sensor' column.")
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
    parser.add_argument("--t0_col", default="path_t0")
    parser.add_argument("--t90_col", default="path_t90")
    parser.add_argument("--t360_col", default="path_t360")
    parser.add_argument(
        "--skip_invalid_samples",
        action="store_true",
        help="Drop CSV rows whose TIFFs are missing or unreadable instead of failing mid-epoch.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Base directory to store checkpoints (latest/best).",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run name for checkpoint subfolder. Defaults to mixed_<train_csv_stem>__<test_csv_stem>__ft|head.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume from the latest checkpoint for this run name.",
    )
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
    parser.add_argument("--local_cache_dir", default=None, help="Directory for caching remote files.")
    parser.add_argument("--local_cache_warmup", action="store_true", help="Pre-copy all files to the cache before training.")
    parser.add_argument("--local_cache_workers", type=int, default=12, help="Number of workers for cache warmup.")
    args = parser.parse_args()
    main(args)
