import argparse
import csv
import hashlib
import sys
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Optional, Sequence

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


def main(args):
    from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset
    
    class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
        def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._local_file_cache = local_file_cache

        def _load_image(self, path: str, *, column_name=None, sample_id=None):
            if self._local_file_cache is not None and isinstance(path, str):
                path = self._local_file_cache.ensure_local(path)
            return super()._load_image(path, column_name=column_name, sample_id=sample_id)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir)

    wv3_band_names = [b.strip() for b in args.wv3_bands.split(",") if b.strip()]
    if len(wv3_band_names) == 0:
        raise ValueError("--wv3_bands must provide at least one WV3 band column name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(
        args.wv3_srf_csv, wv3_band_names, include_sigmas=args.wv3_full_spectra
    )

    # Use three WV3 temporal frames as input: t0, t-90, t-360.
    base_train_ds = CachedS2TemporalCsvDataset(
        csv_path=args.train_csv,
        ds_cfg_name=args.ds_cfg_name,
        normalize_stats=None,  # inputs are already normalized upstream
        scale_to_unit=False,
        pad_to_multiple=args.pad_to_multiple,
        compute_stats=False,
        path_columns=(args.t0_col, args.t90_col, args.t360_col),
        skip_invalid_samples=args.skip_invalid_samples,
        local_file_cache=cache_obj,
    )
    base_test_ds = CachedS2TemporalCsvDataset(
        csv_path=args.test_csv,
        ds_cfg_name=args.ds_cfg_name,
        normalize_stats=None,
        scale_to_unit=False,
        pad_to_multiple=args.pad_to_multiple,
        compute_stats=False,
        path_columns=(args.t0_col, args.t90_col, args.t360_col),
        skip_invalid_samples=args.skip_invalid_samples,
        local_file_cache=cache_obj,
    )
    base_train_ds.chn_ids = wv3_chn_ids
    base_test_ds.chn_ids = wv3_chn_ids.clone()
    wv3_mus = base_train_ds.chn_ids[:, 0] if base_train_ds.chn_ids.ndim == 2 else base_train_ds.chn_ids
    print(
        f"WV3 channel IDs loaded from {args.wv3_srf_csv}: bands/timepoint={len(wv3_mus)}, total_temporal_bands={len(wv3_mus) * 3}, "
        f"mu_range=[{int(wv3_mus.min())}, {int(wv3_mus.max())}] nm",
        flush=True,
    )

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for col in (args.t0_col, args.t90_col, args.t360_col):
            if col in base_train_ds.df.columns:
                all_paths.extend(base_train_ds.df[col].dropna().astype(str).tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    train_ds = ConcatTemporalDataset(base_train_ds)
    test_ds = ConcatTemporalDataset(base_test_ds)

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
        tp = fp = fn = tn = 0
        all_probs = []
        all_targets = []
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
                all_targets.append(labels.detach().cpu())
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

                if args.max_eval_steps is not None and step >= args.max_eval_steps:
                    break

        test_acc = correct / total
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
            f"recall={recall:.4f} fpr={fpr:.4f} auroc={test_auroc:.4f}",
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
        help="CSV column for t0 image path.",
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
        "--skip_invalid_samples",
        action="store_true",
        help="Drop CSV rows whose TIFFs are missing or unreadable instead of failing mid-epoch.",
    )
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=8)
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
    args = parser.parse_args()
    main(args)
