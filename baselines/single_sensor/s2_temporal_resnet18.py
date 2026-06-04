import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Reuse the S2 stats from examples/train_head_s2_temporal.py.
PRECOMPUTED_STATS: Tuple[Sequence[float], Sequence[float]] = (
    [786.128173828125, 1025.8876953125, 1593.730712890625, 2315.26123046875, 2710.462890625, 3115.90087890625, 3289.0830078125, 3465.536376953125, 3495.579833984375, 3517.7958984375, 4180.28564453125, 3567.866943359375],
    [435.72607421875, 597.6113891601562, 688.5059814453125, 840.1614990234375, 801.7208251953125, 706.9466552734375, 689.823974609375, 727.5567626953125, 668.30224609375, 551.3565063476562, 629.679931640625, 641.590087890625],
)


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
        },
    )
    return run


class S2TemporalStackedDataset:
    """
    Minimal temporal dataset that:
    - Reads three Sentinel-2 timepoints from a CSV (t0, t-90, t-360).
    - Normalizes each timepoint with the provided mean/std.
    - Stacks them into a single 36-channel tensor for a plain CNN baseline.
    - Mutes bands that are all-zero before normalization to avoid NaNs/large constants.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        path_columns: Tuple[str, str, str] = ("image_path", "s2_pre_path", "s2_pre_pre_path"),
        ds_cfg_name: str = "s2_12band",
        normalize_stats: Tuple[Sequence[float], Sequence[float]] = PRECOMPUTED_STATS,
        pad_to_multiple: int = 14,
    ):
        from thirdparty.dinov2.data.datasets.s2_csv import S2TemporalCsvDataset

        if normalize_stats is None:
            raise ValueError("normalize_stats must be provided to ensure consistent preprocessing.")

        # Reuse the CSV reading + TIFF loading logic, but keep raw values so we can manually normalize with mute support.
        self._base = S2TemporalCsvDataset(
            csv_path=csv_path,
            ds_cfg_name=ds_cfg_name,
            path_columns=path_columns,
            normalize_stats=None,
            scale_to_unit=False,
            compute_stats=False,
            pad_to_multiple=pad_to_multiple,
            transform=None,
        )
        mean, std = normalize_stats
        mean_tensor = torch.tensor(mean, dtype=torch.float32)
        std_tensor = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
        self._mean = mean_tensor.view(-1, 1, 1)
        self._std = std_tensor.view(-1, 1, 1)
        self._pad_to_multiple = pad_to_multiple
        self._path_columns = path_columns

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        row = self._base.df.iloc[idx]
        label = int(row[self._base.label_column])

        stacked_timepoints = []
        for col in self._path_columns:
            img = self._base._read_image_raw(row[col])  # (12, H, W) float32
            if self._pad_to_multiple is not None:
                img = self._base._pad_to_multiple(img, self._pad_to_multiple)
            zero_mask = (img == 0).view(img.shape[0], -1).all(dim=1)  # older torch lacks tuple dims for all()
            img = (img - self._mean) / self._std
            if zero_mask.any():
                img[zero_mask] = 0.0
            stacked_timepoints.append(img)

        stacked = torch.cat(stacked_timepoints, dim=0)  # (36, H, W)
        return stacked, label


def build_resnet18(num_channels: int = 36, num_classes: int = 2) -> nn.Module:
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
    train_ds = S2TemporalStackedDataset(
        csv_path=args.train_csv,
        path_columns=path_columns,
        ds_cfg_name="s2_12band",
        normalize_stats=PRECOMPUTED_STATS,
        pad_to_multiple=args.pad_to_multiple,
    )
    test_ds = S2TemporalStackedDataset(
        csv_path=args.test_csv,
        path_columns=path_columns,
        ds_cfg_name="s2_12band",
        normalize_stats=PRECOMPUTED_STATS,
        pad_to_multiple=args.pad_to_multiple,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    print(
        f"ResNet-18 baseline with 3x12 channels | train_samples={len(train_ds)} test_samples={len(test_ds)} "
        f"pad_to_multiple={args.pad_to_multiple}",
        flush=True,
    )

    model = build_resnet18(num_channels=36, num_classes=2).to(device)
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
    parser = argparse.ArgumentParser(description="Standalone ResNet-18 baseline for S2 temporal classification (3x12 bands).")
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
    parser.add_argument("--t0_col", default="image_path", help="CSV column for t0 image path.")
    parser.add_argument("--t90_col", default="s2_pre_path", help="CSV column for t-90 image path.")
    parser.add_argument("--t360_col", default="s2_pre_pre_path", help="CSV column for t-360 image path.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    args = parser.parse_args()
    main(args)
