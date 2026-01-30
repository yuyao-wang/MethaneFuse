import argparse
import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Optional, Sequence, Tuple

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Landsat 8/9 SR bands (B1-B7) mean/std computed on training split.
PRECOMPUTED_STATS: Tuple[Sequence[float], Sequence[float]] = (
    [10729.92784546, 11384.64407242, 13172.77519667, 14892.25620267, 18149.92169893, 20249.17615773, 18375.0669698],
    [1029.18232283, 1188.52313418, 1552.27685613, 1959.74400972, 1954.80410093, 2098.98682671, 1895.56781996],
)


class LocalFileCache:
    """
    Simple filesystem cache that mirrors remote TIFFs into a local directory.
    Copies are content-addressed by SHA1 of the absolute source path to avoid collisions.
    Supports an optional soft size cap with LRU eviction.
    """

    def __init__(self, cache_dir: str, max_bytes: Optional[int] = None):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

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

        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
            self._trim_to_size()
        except Exception:
            with suppress(FileNotFoundError):
                tmp.unlink()
            raise
        return str(dst)

    def _trim_to_size(self):
        if self.max_bytes is None:
            return

        total = 0
        files = []
        for sub in self.cache_dir.glob("*/*"):
            if sub.is_file():
                st = sub.stat()
                total += st.st_size
                files.append((st.st_atime, st.st_size, sub))

        if total <= self.max_bytes:
            return

        files.sort(key=lambda x: x[0])  # oldest access first
        for _, size, path in files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            total -= size
            if total <= self.max_bytes:
                break

    def warm_up(self, paths: Sequence[str], max_workers: int = 4) -> None:
        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths:
            return

        print(
            f"[cache] Pre-warming {len(unique_paths)} files into {self.cache_dir} using {max_workers} workers",
            flush=True,
        )

        def _copy_one(path: str):
            try:
                self.ensure_local(path)
                return None
            except Exception as exc:  # pragma: no cover - debug helper
                return path, exc

        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    errors.append(res)

        if errors:
            print(f"[cache] Warning: {len(errors)} files failed to cache (showing up to 5)", flush=True)
            for path, exc in errors[:5]:
                print(f"[cache]   {path}: {exc}", flush=True)


class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
    """
    S2TemporalCsvDataset variant that first copies TIFFs into a local cache directory.
    """

    def __init__(self, *args, local_file_cache: Optional[LocalFileCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            try:
                path = self._local_file_cache.ensure_local(path)
            except FileNotFoundError as exc:
                raise _SkipSample(f"Missing file during cache copy: {path}") from exc
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)

    def warm_up_cache(self, max_workers: int = 4) -> None:
        if self._local_file_cache is None:
            return
        all_paths = []
        for col in self.path_columns:
            all_paths.extend(self.df[col].tolist())
        self._local_file_cache.warm_up(all_paths, max_workers=max_workers)


def init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "pad_to_multiple": args.pad_to_multiple,
            "device": args.device,
            "t0_col": args.t0_col,
            "t90_col": args.t90_col,
            "t360_col": args.t360_col,
            "warmup_steps": args.warmup_steps,
            "skip_invalid_samples": args.skip_invalid_samples,
        },
    )
    return run


class ConcatTemporalDataset(Dataset):
    """
    Wrap the temporal Landsat 8/9 CSV dataset and concatenate three 7-band timepoints into one 21-band tensor.
    Optionally caches remote TIFFs into a local directory to avoid repeated network reads.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        path_columns: Tuple[str, str, str] = ("path_t0", "path_t90", "path_t360"),
        ds_cfg_name: str = "landsat89_7band",
        normalize_stats: Tuple[Sequence[float], Sequence[float]] = PRECOMPUTED_STATS,
        pad_to_multiple: int = 14,
        skip_invalid_samples: bool = False,
        local_cache_dir: Optional[str] = None,
        cache_warmup: bool = False,
        cache_workers: int = 4,
        cache_max_gb: Optional[float] = None,
    ):
        dataset_cls = S2TemporalCsvDataset
        cache_obj = None
        extra_kwargs = {}
        if local_cache_dir:
            dataset_cls = CachedS2TemporalCsvDataset
            max_bytes = None if cache_max_gb is None else int(cache_max_gb * (1024**3))
            cache_obj = LocalFileCache(local_cache_dir, max_bytes=max_bytes)
            extra_kwargs["local_file_cache"] = cache_obj

        self._base = dataset_cls(
            csv_path=csv_path,
            ds_cfg_name=ds_cfg_name,
            normalize_stats=normalize_stats,
            scale_to_unit=False,
            pad_to_multiple=pad_to_multiple,
            compute_stats=False,
            path_columns=path_columns,
            skip_invalid_samples=skip_invalid_samples,
            **extra_kwargs,
        )
        if cache_warmup and cache_obj is not None:
            self._base.warm_up_cache(max_workers=cache_workers)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        x_list, label = self._base[idx]  # list of three dicts with "imgs" tensors
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)  # (21, H, W)
        return imgs, int(label)


def build_resnet18(num_channels: int = 21, num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, correct / total


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

    path_columns = (args.t0_col, args.t90_col, args.t360_col)
    train_ds = ConcatTemporalDataset(
        csv_path=args.train_csv,
        path_columns=path_columns,
        ds_cfg_name="landsat89_7band",
        normalize_stats=PRECOMPUTED_STATS,
        pad_to_multiple=args.pad_to_multiple,
        skip_invalid_samples=args.skip_invalid_samples,
        local_cache_dir=args.local_cache_dir,
        cache_warmup=args.local_cache_warmup,
        cache_workers=args.local_cache_workers,
        cache_max_gb=args.local_cache_max_gb,
    )
    test_ds = ConcatTemporalDataset(
        csv_path=args.test_csv,
        path_columns=path_columns,
        ds_cfg_name="landsat89_7band",
        normalize_stats=PRECOMPUTED_STATS,
        pad_to_multiple=args.pad_to_multiple,
        skip_invalid_samples=args.skip_invalid_samples,
        local_cache_dir=args.local_cache_dir,
        cache_warmup=False,
        cache_workers=args.local_cache_workers,
        cache_max_gb=args.local_cache_max_gb,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    print(
        f"ResNet-18 baseline (L8/9 SR, 3x7 channels) | train_samples={len(train_ds)} test_samples={len(test_ds)} "
        f"pad_to_multiple={args.pad_to_multiple}",
        flush=True,
    )

    model = build_resnet18(num_channels=21, num_classes=2).to(device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
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
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        global_step += len(train_loader)
        print(
            f"Epoch {epoch} done | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}",
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
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone ResNet-18 baseline for Landsat 8/9 temporal classification (3x7 bands).")
    parser.add_argument("--train_csv", default="data_csv/hongxuan_temporal_32/train.csv")
    parser.add_argument("--test_csv", default="data_csv/hongxuan_temporal_32/test.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--device", default="cpu", help='PyTorch device string, e.g. "cpu" or "cuda".')
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N steps.")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Noam scheduler warmup steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument("--t0_col", default="path_t0", help="CSV column for t0 image path.")
    parser.add_argument("--t90_col", default="path_t90", help="CSV column for t-90 image path.")
    parser.add_argument("--t360_col", default="path_t360", help="CSV column for t-360 image path.")
    parser.add_argument(
        "--local_cache_dir",
        default=None,
        help="Directory for caching remote TIFFs locally. Disabled when not set.",
    )
    parser.add_argument(
        "--local_cache_max_gb",
        type=float,
        default=None,
        help="Optional soft cap for cache size in GB; least-recently-used files are evicted when exceeded.",
    )
    parser.add_argument(
        "--local_cache_warmup",
        action="store_true",
        help="When set, pre-copy all referenced TIFFs into the local cache before training starts.",
    )
    parser.add_argument(
        "--local_cache_workers",
        type=int,
        default=4,
        help="Number of worker threads to use while warming the cache.",
    )
    parser.add_argument(
        "--skip_invalid_samples",
        action="store_true",
        help="Drop CSV rows whose TIFFs are missing or unreadable instead of failing mid-epoch.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    args = parser.parse_args()
    main(args)
