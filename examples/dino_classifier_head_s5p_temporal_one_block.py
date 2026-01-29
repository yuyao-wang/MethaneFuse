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

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# Per-timeframe normalization stats for S5P CH4 stacked (t0, t-90, t-360).
PRECOMPUTED_STATS = (
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


class CLSHead(nn.Module):
    """Minimal DINO-style classifier head: CLS -> LayerNorm -> Linear."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls))


class ConcatTemporalDataset(Dataset):
    """Wraps S5pTemporalNpzDataset to concatenate all timepoints along the channel dimension."""

    def __init__(self, base_ds, resize_to: Optional[int] = 224):
        self.base_ds = base_ds
        self.resize_to = resize_to

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_list, label = self.base_ds[idx]
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0)
        chn_ids = torch.cat([x["chn_ids"] for x in x_list], dim=0)
        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        if self.resize_to is not None:
            x_dict = resize_imgs(x_dict, self.resize_to)
        return x_dict, label


def load_backbone(weights_path: str, device: torch.device, debug: bool = False):
    """Build the Panopticon ViT backbone and optionally load pretrained weights."""

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


def resize_imgs(x_dict, size: int):
    """Resize x_dict["imgs"] to (size, size) spatial resolution via bilinear upsampling."""

    imgs = x_dict.get("imgs")
    if imgs is None:
        return x_dict

    squeeze_back = False
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
        squeeze_back = True

    if imgs.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor for imgs, got shape {tuple(imgs.shape)}")

    imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    if squeeze_back:
        imgs = imgs.squeeze(0)
    x_dict["imgs"] = imgs
    return x_dict


def set_trainable(module: nn.Module, requires_grad: bool):
    if isinstance(module, nn.DataParallel):
        module = module.module
    for p in module.parameters():
        p.requires_grad = requires_grad


def maybe_wrap_dataparallel(module: nn.Module, use_dp: bool, device: torch.device, name: str) -> nn.Module:
    if not use_dp:
        return module
    if device.type != "cuda":
        print(f"[DataParallel:{name}] Requested but device={device} is not CUDA; running single-device.", flush=True)
        return module
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        print(
            f"[DataParallel:{name}] Requested but only {gpu_count} CUDA device detected; running single-device.",
            flush=True,
        )
        return module
    print(f"Enabling DataParallel for {name} across {gpu_count} GPUs", flush=True)
    return nn.DataParallel(module)


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
    from dinov2.data.datasets.s5p_npz import S5pTemporalNpzDataset

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    norm_stats = load_normalization_stats(args)
    chn_ids = parse_comma_separated_floats(args.chn_ids)
    stats_subset = parse_subset_value(args.compute_stats_subset)
    path_columns = (args.t0_col,) if args.stacked_time_npz else (args.t0_col, args.t90_col, args.t360_col)

    base_train_ds = S5pTemporalNpzDataset(
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
        path_columns=path_columns,
        data_key=args.data_key,
        chn_ids_key=args.chn_ids_key,
        channels_first=not args.channel_last,
        default_chn_id_value=args.default_chn_id_value,
        stacked_time=args.stacked_time_npz,
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

    base_test_ds = S5pTemporalNpzDataset(
        csv_path=args.test_csv,
        ds_cfg_name=args.ds_cfg_name,
        chn_ids=chn_ids,
        normalize_stats=resolved_stats,
        scale_to_unit=args.scale_to_unit,
        scale_value=args.scale_value,
        compute_stats=False,
        pad_to_multiple=args.pad_to_multiple,
        pad_value=args.pad_value,
        path_columns=path_columns,
        data_key=args.data_key,
        chn_ids_key=args.chn_ids_key,
        channels_first=not args.channel_last,
        default_chn_id_value=args.default_chn_id_value,
        stacked_time=args.stacked_time_npz,
        allow_pickle=args.allow_pickle,
        nan_to_num=args.nan_to_num,
    )

    train_ds = ConcatTemporalDataset(base_train_ds, resize_to=args.resize_size)
    test_ds = ConcatTemporalDataset(base_test_ds, resize_to=args.resize_size)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    stats_status = base_train_ds.get_stats_source()
    if stats_status is None:
        stats_status = "provided" if resolved_stats is not None else "none"

    print(
        f"Using device={device}, resize={args.resize_size}, train_samples={len(train_ds)}, test_samples={len(test_ds)}, "
        f"norm_stats={stats_status}, train_backbone={args.train_backbone}",
        flush=True,
    )

    backbone = load_backbone(args.weights, device=device, debug=args.debug).to(device)
    head = CLSHead(embed_dim=args.embed_dim, num_classes=2).to(device)

    data_parallel = args.data_parallel
    backbone = maybe_wrap_dataparallel(backbone, data_parallel, device, "backbone")
    head = maybe_wrap_dataparallel(head, data_parallel, device, "head")

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
        freeze_backbone = (not will_train_backbone) or (
            args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        )
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
                        {
                            "train_loss_running": running_loss,
                            "train_acc_running": running_acc,
                            "step": global_step,
                            "epoch": epoch,
                        }
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
        description="Panopticon ViT + CLS head for Sentinel-5P temporal inputs concatenated along channels."
    )
    parser.add_argument("--train_csv", default="data_csv/hongxuan_temporal_32/train.csv")
    parser.add_argument("--test_csv", default="data_csv/hongxuan_temporal_32/test.csv")
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
    parser.add_argument("--pad_to_multiple", type=int, default=None, help="Pad H/W to this multiple (set None to skip).")
    parser.add_argument("--pad_value", type=float, default=0.0, help="Pad value when extending to pad_to_multiple.")
    parser.add_argument("--scale_to_unit", action="store_true", help="Divide NPZ values by scale_value before normalization.")
    parser.add_argument("--scale_value", type=float, default=65535.0, help="Divisor used when scale_to_unit is enabled.")
    parser.add_argument("--data_key", default=None, help="Optional NPZ key that stores the image array (defaults to first).")
    parser.add_argument("--chn_ids_key", default="chn_ids", help="NPZ key containing per-sample channel IDs if available.")
    parser.add_argument("--channel_last", action="store_true", help="Set if NPZ arrays are stored as HWC instead of CHW.")
    parser.add_argument("--ds_cfg_name", default=None, help="Optional dataset config name for channel IDs.")
    parser.add_argument("--chn_ids", default=None, help="Comma-separated channel IDs to override ds_cfg_name/NPZ.")
    parser.add_argument(
        "--default_chn_id_value",
        type=float,
        default=0.0,
        help="Constant channel id used when none are provided (keeps model happy for single-channel grids).",
    )
    parser.add_argument(
        "--stacked_time_npz",
        action="store_true",
        help="CSV has a single NPZ path containing stacked time slices along the first/channel dimension (e.g., shape (3,H,W)).",
    )
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow loading NPZ files that contain pickled data (enable only if you trust the source).",
    )
    parser.add_argument(
        "--nan_to_num",
        type=float,
        default=None,
        help="If set, replace NaN/Inf in NPZ arrays with this value before normalization.",
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
    parser.add_argument("--resize_size", type=int, default=224, help="Resize spatial resolution before feeding the backbone.")
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
        default="image_path_224",
        help="CSV column for t0 image path (or stacked NPZ path when stacked_time_npz is enabled).",
    )
    parser.add_argument(
        "--t90_col",
        default="s5p_pre_path",
        help="CSV column for t-90 image path (ignored if stacked_time_npz).",
    )
    parser.add_argument(
        "--t360_col",
        default="s5p_pre_pre_path",
        help="CSV column for t-360 image path (ignored if stacked_time_npz).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Wrap backbone and head with torch.nn.DataParallel when multiple CUDA GPUs are available.",
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
    args = parser.parse_args()
    main(args)
