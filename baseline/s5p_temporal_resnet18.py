import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Reuse the Sentinel-5P stats from examples/train_head_s5p_temporal.py (t0, t-90, t-360).
PRECOMPUTED_STATS: Tuple[Sequence[float], Sequence[float]] = (
    [1908.944798144908, 362.20128748570943, 666.5645739544017],
    [52.2615669211965, 743.7553740768346, 901.8399543163595],
)


def parse_comma_separated_floats(val: Optional[str]) -> Optional[Sequence[float]]:
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse floats from '{val}'") from exc


def parse_subset_value(val: Optional[str]) -> Optional[Union[int, float]]:
    if val is None:
        return None
    try:
        text = str(val)
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("compute_stats_subset must be an int or float") from exc


def load_normalization_stats(args) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    if args.normalize_stats_npz is not None:
        with np.load(args.normalize_stats_npz) as npz:
            if "mean" not in npz or "std" not in npz:
                raise KeyError(f"{args.normalize_stats_npz} must contain 'mean' and 'std'")
            mean = npz["mean"].tolist()
            std = npz["std"].tolist()
            return mean, std

    mean_override = parse_comma_separated_floats(args.normalize_mean)
    std_override = parse_comma_separated_floats(args.normalize_std)
    if mean_override is not None and std_override is not None:
        if len(mean_override) != len(std_override):
            raise ValueError("normalize_mean and normalize_std must have the same length")
        return mean_override, std_override
    if (mean_override is None) != (std_override is None):
        raise ValueError("normalize_mean and normalize_std must be provided together")

    return PRECOMPUTED_STATS


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
            # "t90_col": args.t90_col,
            # "t360_col": args.t360_col,
            "resize_size": args.resize_size,
            "warmup_steps": args.warmup_steps,
            "ds_cfg_name": args.ds_cfg_name,
            "data_key": args.data_key,
            "chn_ids_key": args.chn_ids_key,
            "channel_last": args.channel_last,
            # "stacked_time_npz": args.stacked_time_npz,
        },
    )
    return run


class S5pSimpleNpzDataset(Dataset):
    """Wraps S5pNpzDataset to return a single tensor for CNNs."""

    def __init__(self, base_ds, resize_size: Optional[int] = None):
        self.base_ds = base_ds
        self.resize_size = resize_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_dict, label = self.base_ds[idx]
        imgs = x_dict["imgs"]  # (C, H, W)
        if self.resize_size is not None:
            imgs = _resize_tensor(imgs, self.resize_size)
        return imgs, label

def _resize_tensor(imgs: torch.Tensor, size: int) -> torch.Tensor:
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
        squeeze_back = True
    elif imgs.ndim == 4:
        squeeze_back = False
    else:
        raise ValueError(f"Expected 3D/4D tensor for resizing, got {tuple(imgs.shape)}")

    imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    if squeeze_back:
        imgs = imgs.squeeze(0)
    return imgs


def build_resnet18(num_channels: int, num_classes: int = 2) -> nn.Module:
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


def infer_num_channels(dataset: Dataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot infer channel count.")
    sample, _ = dataset[0]
    if sample.ndim != 3:
        raise ValueError(f"Expected CHW tensor from dataset, got shape {tuple(sample.shape)}")
    return sample.shape[0]


def main(args):
    from dinov2.data.datasets.s5p_npz import S5pNpzDataset

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

    norm_stats = load_normalization_stats(args)
    chn_ids = parse_comma_separated_floats(args.chn_ids)
    stats_subset = parse_subset_value(args.compute_stats_subset)

    base_train_ds = S5pNpzDataset(
        csv_path=args.train_csv,
        ds_cfg_name=args.ds_cfg_name,
        chn_ids=chn_ids,
        normalize_stats=norm_stats,
        scale_to_unit=args.scale_to_unit,
        scale_value=args.scale_value,
        compute_stats=args.compute_stats and norm_stats is None,
        compute_stats_subset=stats_subset,
        pad_to_multiple=args.pad_to_multiple,
        pad_value=args.pad_value,
        path_column=args.t0_col,
        data_key=args.data_key,
        chn_ids_key=args.chn_ids_key,
        channels_first=not args.channel_last,
        default_chn_id_value=args.default_chn_id_value,
        allow_pickle=args.allow_pickle,
        nan_to_num=args.nan_to_num,
    )

    resolved_stats = base_train_ds.get_normalize_stats()
    if resolved_stats is None:
        resolved_stats = norm_stats
    elif norm_stats is None:
        mean, std = resolved_stats
        print(
            f"Computed normalization stats from training set (mean len={len(mean)}, std len={len(std)})",
            flush=True,
        )

    base_test_ds = S5pNpzDataset(
        csv_path=args.test_csv,
        ds_cfg_name=args.ds_cfg_name,
        chn_ids=chn_ids,
        normalize_stats=resolved_stats,
        scale_to_unit=args.scale_to_unit,
        scale_value=args.scale_value,
        compute_stats=False,
        pad_to_multiple=args.pad_to_multiple,
        pad_value=args.pad_value,
        path_column=args.t0_col,
        data_key=args.data_key,
        chn_ids_key=args.chn_ids_key,
        channels_first=not args.channel_last,
        default_chn_id_value=args.default_chn_id_value,
        allow_pickle=args.allow_pickle,
        nan_to_num=args.nan_to_num,
    )

    train_ds = S5pSimpleNpzDataset(base_train_ds, resize_size=args.resize_size)
    test_ds = S5pSimpleNpzDataset(base_test_ds, resize_size=args.resize_size)

    stats_status = base_train_ds.get_stats_source()
    if stats_status is None:
        stats_status = "provided" if resolved_stats is not None else "none"

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    num_channels = infer_num_channels(train_ds)
    print(
        f"S5P ResNet-18 baseline with {num_channels} input channels | "
        f"train_samples={len(train_ds)} test_samples={len(test_ds)} pad_to_multiple={args.pad_to_multiple} "
        f"resize={args.resize_size} norm_stats={stats_status}",
        flush=True,
    )

    model = build_resnet18(num_channels=num_channels, num_classes=2).to(device)
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
    parser = argparse.ArgumentParser(
        description="Standalone ResNet-18 baseline for Sentinel-5P temporal classification (stacked 3 timepoints)."
    )
    parser.add_argument("--train_csv", default="data_csv/hongxuan_temporal_32/train.csv")
    parser.add_argument("--test_csv", default="data_csv/hongxuan_temporal_32/test.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--pad_value", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", help='PyTorch device string, e.g. "cpu" or "cuda".')
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N steps.")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Noam scheduler warmup steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument("--resize_size", type=int, default=224, help="Spatial resize before feeding ResNet-18.")
    parser.add_argument("--t0_col", default="image_path", help="CSV column for the NPZ path.")
    # parser.add_argument(
    #     "--t90_col",
    #     default="s5p_pre_path",
    #     help="CSV column for t-90 image path (only used when --separate_time_paths is set).",
    # )
    # parser.add_argument(
    #     "--t360_col",
    #     default="s5p_pre_pre_path",
    #     help="CSV column for t-360 image path (only used when --separate_time_paths is set).",
    # )
    parser.add_argument("--ds_cfg_name", default=None, help="Optional dataset config name for channel IDs.")
    parser.add_argument("--chn_ids", default=None, help="Comma-separated channel IDs to override ds_cfg_name/NPZ.")
    parser.add_argument("--data_key", default=None, help="Optional NPZ key storing the image array (defaults to first entry).")
    parser.add_argument("--chn_ids_key", default="chn_ids", help="NPZ key containing per-sample channel IDs if available.")
    parser.add_argument("--channel_last", action="store_true", help="Set if NPZ arrays are stored as HWC instead of CHW.")
    parser.add_argument("--allow_pickle", action="store_true", help="Allow loading NPZ files containing pickled data.")
    parser.add_argument("--nan_to_num", type=float, default=None, help="Replace NaN/Inf in NPZ arrays with this constant.")
    parser.add_argument("--scale_to_unit", action="store_true", help="Divide NPZ values by scale_value when no stats are provided.")
    parser.add_argument("--scale_value", type=float, default=65535.0, help="Divisor used when scale_to_unit is enabled.")
    parser.add_argument(
        "--default_chn_id_value",
        type=float,
        default=0.0,
        help="Fallback channel id used when none are provided.",
    )
    parser.add_argument(
        "--normalize_stats_npz",
        default=None,
        help="Optional .npz file with 'mean' and 'std' arrays to use for normalization.",
    )
    parser.add_argument("--normalize_mean", default=None, help="Override normalization mean as comma-separated floats.")
    parser.add_argument("--normalize_std", default=None, help="Override normalization std as comma-separated floats.")
    parser.add_argument(
        "--compute_stats_subset",
        default=None,
        help="Limit samples when computing normalization stats (int count or float fraction).",
    )
    parser.add_argument(
        "--compute_stats",
        dest="compute_stats",
        action="store_true",
        help="Compute per-channel mean/std from the training set when stats are not provided.",
    )
    parser.add_argument("--no_compute_stats", dest="compute_stats", action="store_false", help="Skip computing stats.")
    parser.set_defaults(compute_stats=True)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    args = parser.parse_args()
    main(args)
