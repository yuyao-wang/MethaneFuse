import argparse
import hashlib
import os
import random
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
from torchvision import models
from tqdm import tqdm

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

try:
    from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample
except ImportError:  # Fallback stubs to keep the script importable
    class _SkipSample(Exception):
        pass

    S2TemporalCsvDataset = object  # type: ignore

# Default statistics used as a fallback when no valid pixels are found.
DEFAULT_STATS = (
    [10000.0] * 21,
    [2000.0] * 21,
)


class StaticAnchoredCache:
    """Optional local file cache to speed up repeated TIFF loading."""

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
        print(f"[Cache] Warmup {len(unique_paths)} files...", flush=True)

        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            fallback_count = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"[Cache] Done. Cached: {len(unique_paths) - fallback_count}, remote: {fallback_count}")


class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            path = self._local_file_cache.ensure_local(path)
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)


class ConcatTemporalDataset(Dataset):
    """
    Landsat 8/9 temporal dataset that:
    - Reads three timepoints (t0, t-90, t-360) with 7 bands each.
    - Defers normalization until stats are computed to allow zero-masking.
    - Concatenates into a 21-channel tensor.
    """

    def __init__(self, csv_path: str, **kwargs):
        local_cache_dir = kwargs.pop("local_cache_dir", None)
        cache_warmup = kwargs.pop("cache_warmup", False)
        cache_workers = kwargs.pop("cache_workers", 8)
        cache_min_free_gb = kwargs.pop("cache_min_free_gb", 10.0)

        # Disable internal normalization; we handle it after computing stats.
        kwargs["normalize_stats"] = None

        dataset_cls = S2TemporalCsvDataset
        self.cache_obj = None
        if local_cache_dir:
            dataset_cls = CachedS2TemporalCsvDataset
            self.cache_obj = StaticAnchoredCache(local_cache_dir, min_free_gb=cache_min_free_gb)
            kwargs["local_file_cache"] = self.cache_obj

        self._base = dataset_cls(csv_path=csv_path, **kwargs)

        if cache_warmup and self.cache_obj:
            all_paths = []
            for col in self._base.path_columns:
                all_paths.extend(self._base.df[col].tolist())
            self.cache_obj.warm_up(all_paths, max_workers=cache_workers)

        self._mean_tensor: Optional[torch.Tensor] = None
        self._std_tensor: Optional[torch.Tensor] = None

    def update_stats(self, mean: torch.Tensor, std: torch.Tensor):
        std = torch.clamp(std, min=1e-6)
        self._mean_tensor = mean.view(-1, 1, 1)
        self._std_tensor = std.view(-1, 1, 1)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        x_list, label = self._base[idx]
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)  # (21, H, W)
        if self._mean_tensor is not None and self._std_tensor is not None:
            mask = imgs == 0
            imgs = (imgs - self._mean_tensor) / self._std_tensor
            imgs[mask] = 0.0
        return imgs, int(label)


def compute_dynamic_stats(dataset: Dataset, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample images, remove 0 (No-Data) pixels, and compute per-channel mean/std.
    """

    print(f"[Stats] Sampling {n_samples} examples to compute mean/std (zeros ignored)...", flush=True)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    channel_sums = torch.zeros(21)
    channel_sq_sums = torch.zeros(21)
    channel_counts = torch.zeros(21)

    for i in tqdm(indices, desc="Sampling"):
        try:
            img, _ = dataset[i]  # (21, H, W)
            img = img.float()
            for c in range(21):
                data = img[c]
                valid = data[data != 0]
                if valid.numel() > 0:
                    channel_sums[c] += valid.sum()
                    channel_sq_sums[c] += (valid**2).sum()
                    channel_counts[c] += valid.numel()
        except Exception:
            continue

    means = channel_sums / (channel_counts + 1e-6)
    variances = (channel_sq_sums / (channel_counts + 1e-6)) - means**2
    stds = torch.sqrt(torch.clamp(variances, min=1e-6))

    if channel_counts.min() == 0:
        print("[Stats] Warning: some channels had no valid pixels; filled with defaults.")
        for c in range(21):
            if channel_counts[c] == 0:
                means[c], stds[c] = DEFAULT_STATS[0][c], DEFAULT_STATS[1][c]

    print(f"[Stats] Done. mean[:3]={means[:3].tolist()}, std[:3]={stds[:3].tolist()}")
    return means, stds


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))


def build_vit(num_channels: int = 21, num_classes: int = 2, image_size: int = 224) -> nn.Module:
    model = models.vit_b_16(weights=None, image_size=image_size)
    model.conv_proj = nn.Conv2d(num_channels, model.conv_proj.out_channels, kernel_size=16, stride=16)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    log_interval: int = 50,
    *,
    wandb_run=None,
    epoch: int = 0,
    global_step_start: int = 0,
    scheduler=None,
):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device)
        if images.shape[-1] != model.image_size or images.shape[-2] != model.image_size:
            images = F.interpolate(images, size=(model.image_size, model.image_size), mode="bilinear", align_corners=False)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        if log_interval and step % log_interval == 0:
            running_loss = total_loss / total
            running_acc = correct / total
            print(f"[train] step {step}/{len(loader)} loss={running_loss:.4f} acc={running_acc:.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train_loss_running": running_loss,
                        "train_acc_running": running_acc,
                        "step": global_step_start + step,
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = tn = 0
    all_probs = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        if images.shape[-1] != model.image_size or images.shape[-2] != model.image_size:
            images = F.interpolate(images, size=(model.image_size, model.image_size), mode="bilinear", align_corners=False)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

        all_probs.append(probs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    test_loss = total_loss / total if total > 0 else float("nan")
    test_acc = correct / total if total > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    all_probs = torch.cat(all_probs) if len(all_probs) > 0 else torch.tensor([])
    all_targets = torch.cat(all_targets) if len(all_targets) > 0 else torch.tensor([])
    if all_probs.numel() > 0 and all_targets.numel() > 0:
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(all_targets.numpy(), all_probs.numpy()))
        except Exception:
            auroc = float("nan")
    else:
        auroc = float("nan")

    return test_loss, test_acc, recall, fpr, auroc


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    ds_kwargs = {
        "path_columns": (args.t0_col, args.t90_col, args.t360_col),
        "ds_cfg_name": "landsat89_7band",
        "pad_to_multiple": args.pad_to_multiple,
        "skip_invalid_samples": args.skip_invalid_samples,
        "local_cache_dir": args.local_cache_dir,
        "cache_workers": args.local_cache_workers,
        "cache_min_free_gb": args.cache_min_free_gb,
    }

    train_ds = ConcatTemporalDataset(csv_path=args.train_csv, cache_warmup=args.local_cache_warmup, **ds_kwargs)
    mean, std = compute_dynamic_stats(train_ds, n_samples=args.stats_samples)
    train_ds.update_stats(mean, std)

    test_ds = ConcatTemporalDataset(csv_path=args.test_csv, cache_warmup=False, **ds_kwargs)
    test_ds.update_stats(mean, std)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    print(
        f"ViT-B/16 baseline (L89 3x7 bands) | train_samples={len(train_ds)} test_samples={len(test_ds)} "
        f"pad_to_multiple={args.pad_to_multiple} image_size={args.image_size}",
        flush=True,
    )

    if args.image_size % 16 != 0:
        raise ValueError(f"image_size must be a multiple of 16 (patch size); got {args.image_size}")

    model = build_vit(num_channels=21, num_classes=2, image_size=args.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(args.warmup_steps, 1)
    noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    wandb_run = init_wandb(args)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch} started", flush=True)
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_interval=args.log_interval,
            wandb_run=wandb_run,
            epoch=epoch,
            global_step_start=global_step,
            scheduler=scheduler,
        )
        test_loss, test_acc, recall, fpr, auroc = evaluate(model, test_loader, criterion, device)
        global_step += len(train_loader)
        print(
            f"Epoch {epoch} done | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} recall={recall:.4f} fpr={fpr:.4f} auroc={auroc:.4f}",
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
                    "test_auroc": auroc,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT-B/16 baseline for Landsat 8/9 temporal classification (3x7 bands).")
    parser.add_argument("--train_csv", default="data_csv/train.csv")
    parser.add_argument("--test_csv", default="data_csv/test.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=12)
    parser.add_argument(
        "--cache_min_free_gb",
        type=float,
        default=10.0,
        help="Minimum free space to keep on disk when caching (GB). Cache falls back to source files if below.",
    )
    parser.add_argument("--stats_samples", type=int, default=1000, help="Number of samples for computing mean/std.")
    parser.add_argument("--skip_invalid_samples", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="baselines")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--t0_col", default="path_t0")
    parser.add_argument("--t90_col", default="path_t90")
    parser.add_argument("--t360_col", default="path_t360")
    args = parser.parse_args()
    main(args)
