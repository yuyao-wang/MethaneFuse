import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Reuse the Sentinel-5P stats from examples/train_head_s5p_temporal.py (t0, t-90, t-360).
PRECOMPUTED_STATS: Tuple[Sequence[float], Sequence[float]] = (
    [1908.944798144908, 362.20128748570943, 666.5645739544017],
    [52.2615669211965, 743.7553740768346, 901.8399543163595],
)

INPUT_MODES = ("raw", "current", "residual", "current_residual")
T0_INDEX = 0
PREV1_INDEX = 1
SEASONAL_INDEX = 4


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
            "input_mode": args.input_mode,
            "seed": args.seed,
            # "stacked_time_npz": args.stacked_time_npz,
        },
    )
    return run


class S5pSimpleNpzDataset(Dataset):
    """Wrap S5pNpzDataset and form CNN channels after normalization."""

    def __init__(
        self,
        base_ds,
        resize_size: Optional[int] = None,
        input_mode: str = "raw",
    ):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode={input_mode!r}; expected one of {INPUT_MODES}")
        self.base_ds = base_ds
        self.resize_size = resize_size
        self.input_mode = input_mode

    def __len__(self):
        return len(self.base_ds)

    def _transform_time_channels(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim != 3:
            raise ValueError(f"Expected normalized CHW tensor, got shape {tuple(imgs.shape)}")
        if self.input_mode == "raw":
            return imgs

        if imgs.shape[0] <= T0_INDEX:
            raise ValueError("S5P input has no t0 channel")
        current = imgs[T0_INDEX : T0_INDEX + 1]
        if self.input_mode == "current":
            return current

        required_channels = SEASONAL_INDEX + 1
        if imgs.shape[0] < required_channels:
            raise ValueError(
                f"input_mode={self.input_mode!r} requires at least {required_channels} "
                "normalized time channels ordered as "
                "(t0, prev1, prev2, prev3, seasonal, year); "
                f"received shape {tuple(imgs.shape)}"
            )
        residuals = torch.cat(
            (
                current - imgs[PREV1_INDEX : PREV1_INDEX + 1],
                current - imgs[SEASONAL_INDEX : SEASONAL_INDEX + 1],
            ),
            dim=0,
        )
        if self.input_mode == "residual":
            return residuals
        return torch.cat((current, residuals), dim=0)

    def __getitem__(self, idx):
        x_dict, label = self.base_ds[idx]
        imgs = self._transform_time_channels(x_dict["imgs"])
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


def compute_binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if targets.size != probabilities.size:
        raise ValueError(
            f"Metric target/probability length mismatch: {targets.size} vs {probabilities.size}"
        )

    empty_metrics = {
        "acc": float("nan"),
        "f1": float("nan"),
        "macro_f1": float("nan"),
        "recall": float("nan"),
        "fpr": float("nan"),
        "ap": float("nan"),
        "auroc": float("nan"),
        "best_f1": float("nan"),
        "best_f1_threshold": float("nan"),
        "recall_at_fpr_01": float("nan"),
        "recall_at_fpr_05": float("nan"),
    }
    if targets.size == 0:
        return empty_metrics

    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    metrics = {
        "acc": float(np.mean(predictions == targets)),
        "f1": float(f1_score(targets, predictions, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "ap": float("nan"),
        "auroc": float("nan"),
        "best_f1": float("nan"),
        "best_f1_threshold": float("nan"),
        "recall_at_fpr_01": float("nan"),
        "recall_at_fpr_05": float("nan"),
    }

    try:
        metrics["ap"] = float(average_precision_score(targets, probabilities))
    except ValueError:
        pass

    try:
        precisions, recalls, thresholds = precision_recall_curve(targets, probabilities)
        if thresholds.size:
            f1_curve = np.divide(
                2.0 * precisions[:-1] * recalls[:-1],
                precisions[:-1] + recalls[:-1],
                out=np.zeros_like(precisions[:-1]),
                where=(precisions[:-1] + recalls[:-1]) > 0,
            )
            best_index = int(np.nanargmax(f1_curve))
            metrics["best_f1"] = float(f1_curve[best_index])
            metrics["best_f1_threshold"] = float(thresholds[best_index])
        else:
            metrics["best_f1"] = metrics["f1"]
            metrics["best_f1_threshold"] = 0.5
    except ValueError:
        pass

    if np.unique(targets).size == 2:
        try:
            metrics["auroc"] = float(roc_auc_score(targets, probabilities))
            roc_fpr, roc_tpr, _ = roc_curve(targets, probabilities)
            within_01 = roc_tpr[roc_fpr <= 0.01]
            within_05 = roc_tpr[roc_fpr <= 0.05]
            metrics["recall_at_fpr_01"] = float(within_01.max()) if within_01.size else 0.0
            metrics["recall_at_fpr_05"] = float(within_05.max()) if within_05.size else 0.0
        except ValueError:
            pass
    return metrics


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix}_loss={metrics['loss']:.4f} {prefix}_acc={metrics['acc']:.4f} "
        f"{prefix}_f1={metrics['f1']:.4f} {prefix}_macro_f1={metrics['macro_f1']:.4f} "
        f"{prefix}_ap={metrics['ap']:.4f} {prefix}_auroc={metrics['auroc']:.4f} "
        f"{prefix}_best_f1={metrics['best_f1']:.4f}"
        f"@{metrics['best_f1_threshold']:.4f} "
        f"{prefix}_recall={metrics['recall']:.4f} {prefix}_fpr={metrics['fpr']:.4f} "
        f"{prefix}_recall@fpr1%={metrics['recall_at_fpr_01']:.4f} "
        f"{prefix}_recall@fpr5%={metrics['recall_at_fpr_05']:.4f}"
    )


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
    all_probabilities = []
    all_targets = []

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
        probabilities = F.softmax(logits.detach(), dim=1)[:, 1]
        all_probabilities.append(probabilities.cpu())
        all_targets.append(labels.detach().cpu())

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

    if total == 0:
        metrics = compute_binary_metrics(np.array([]), np.array([]))
        metrics["loss"] = float("nan")
        return metrics
    targets = torch.cat(all_targets).numpy()
    probabilities = torch.cat(all_probabilities).numpy()
    metrics = compute_binary_metrics(targets, probabilities)
    metrics["loss"] = float(total_loss / total)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_probabilities = []
    all_targets = []

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
        probabilities = F.softmax(logits, dim=1)[:, 1]
        all_probabilities.append(probabilities.detach().cpu())
        all_targets.append(labels.detach().cpu())

    if total == 0:
        metrics = compute_binary_metrics(np.array([]), np.array([]))
        metrics["loss"] = float("nan")
        return metrics
    targets = torch.cat(all_targets).numpy()
    probabilities = torch.cat(all_probabilities).numpy()
    metrics = compute_binary_metrics(targets, probabilities)
    metrics["loss"] = float(total_loss / total)
    return metrics


def infer_num_channels(dataset: Dataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot infer channel count.")
    sample, _ = dataset[0]
    if sample.ndim != 3:
        raise ValueError(f"Expected CHW tensor from dataset, got shape {tuple(sample.shape)}")
    return sample.shape[0]


def seed_data_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def default_run_name(args) -> str:
    return (
        f"{Path(args.train_csv).stem}__{Path(args.test_csv).stem}"
        f"__resnet18__{args.input_mode}"
    )


def atomic_write_metrics(path: Path, metrics_history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    try:
        temporary_path.write_text(
            json.dumps(metrics_history, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_save_checkpoint(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(args):
    from thirdparty.dinov2.data.datasets.s5p_npz import S5pNpzDataset

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = args.seed is None
        torch.backends.cudnn.deterministic = args.seed is not None
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
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

    train_ds = S5pSimpleNpzDataset(
        base_train_ds,
        resize_size=args.resize_size,
        input_mode=args.input_mode,
    )
    test_ds = S5pSimpleNpzDataset(
        base_test_ds,
        resize_size=args.resize_size,
        input_mode=args.input_mode,
    )

    stats_status = base_train_ds.get_stats_source()
    if stats_status is None:
        stats_status = "provided" if resolved_stats is not None else "none"

    pin_memory = device.type == "cuda"
    data_generator = None
    worker_init_fn = None
    if args.seed is not None:
        data_generator = torch.Generator()
        data_generator.manual_seed(args.seed)
        worker_init_fn = seed_data_worker
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=data_generator,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    num_channels = infer_num_channels(train_ds)
    print(
        f"S5P ResNet-18 baseline with {num_channels} input channels | "
        f"train_samples={len(train_ds)} test_samples={len(test_ds)} pad_to_multiple={args.pad_to_multiple} "
        f"resize={args.resize_size} input_mode={args.input_mode} norm_stats={stats_status}",
        flush=True,
    )

    model = build_resnet18(num_channels=num_channels, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(args.warmup_steps, 1)
    noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    wandb_run = init_wandb(args)

    run_name = args.run_name or default_run_name(args)
    run_dir = Path(args.checkpoint_dir).expanduser() / run_name
    metrics_path = run_dir / "metrics_history.json"
    best_val_ap_path = run_dir / "ckpt_best_val_ap.pth"
    metrics_history = []
    best_val_ap = float("-inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch} started", flush=True)
        train_metrics = train_one_epoch(
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
        val_metrics = evaluate(model, test_loader, criterion, device)
        global_step += len(train_loader)
        print(
            f"Epoch {epoch} done | {format_metrics('train', train_metrics)} | "
            f"{format_metrics('val', val_metrics)}",
            flush=True,
        )

        epoch_metrics = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "input_mode": args.input_mode,
            "num_input_channels": int(num_channels),
        }
        epoch_metrics.update(
            {f"train_{key}": float(value) for key, value in train_metrics.items()}
        )
        epoch_metrics.update(
            {f"val_{key}": float(value) for key, value in val_metrics.items()}
        )
        metrics_history.append(epoch_metrics)
        atomic_write_metrics(metrics_path, metrics_history)

        val_ap = val_metrics["ap"]
        if np.isfinite(val_ap) and val_ap > best_val_ap:
            best_val_ap = float(val_ap)
            if not args.no_save_checkpoints:
                atomic_save_checkpoint(
                    best_val_ap_path,
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_ap": best_val_ap,
                        "num_input_channels": int(num_channels),
                        "input_mode": args.input_mode,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                )
                print(
                    f"Saved best val AP checkpoint: {best_val_ap_path} "
                    f"(epoch={epoch}, val_ap={best_val_ap:.4f})",
                    flush=True,
                )
        if wandb_run is not None:
            wandb_run.log(epoch_metrics)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone ResNet-18 baseline for stacked-time Sentinel-5P classification."
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
    parser.add_argument(
        "--input_mode",
        choices=INPUT_MODES,
        default="raw",
        help=(
            "raw keeps all stacked channels; current keeps t0; residual uses "
            "normalized t0-prev1 and t0-seasonal; current_residual concatenates "
            "t0 with those two residuals."
        ),
    )
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
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Base directory for metrics and best-validation-AP checkpoint.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Checkpoint subdirectory name; defaults to a manifest/mode-derived name.",
    )
    parser.add_argument(
        "--no_save_checkpoints",
        action="store_true",
        help="Write metrics_history.json but do not save model checkpoints.",
    )
    args = parser.parse_args()
    main(args)
