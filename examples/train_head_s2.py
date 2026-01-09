import argparse
import os
from pathlib import Path
import sys
from typing import Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

PRECOMPUTED_STATS = (
    [0.029740596190094948, 0.03309573978185654, 0.03938313201069832, 0.04678337648510933, 0.05201242119073868, 0.05771360546350479, 0.06091780215501785, 0.06377039849758148, 0.0, 0.0, 0.07156942784786224, 0.06320085376501083],
    [0.011487055569887161, 0.012671221047639847, 0.014261508360505104, 0.018188122659921646, 0.018463384360074997, 0.016438089311122894, 0.016748901456594467, 0.01686006784439087, 0.0, 0.0, 0.02173806168138981, 0.022262444719672203],
)


class MLPHead(nn.Module):
    """Three-layer MLP head for binary classification."""

    def __init__(self, in_dim: int = 768, hidden: Tuple[int, int] = (1024, 512), num_classes: int = 2, drop: float = 0.1):
        super().__init__()
        dims = (in_dim, *hidden, num_classes)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_backbone(weights_path: str, device: torch.device, debug: bool = False):
    from hubconf import _panopticon_vitb14

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


def to_device_batch(x_dict, device):
    return {k: v.to(device) for k, v in x_dict.items()}


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
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "pad_to_multiple": args.pad_to_multiple,
            "device": args.device,
        },
    )
    return run


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        # Default to CUDA:0 when user passes "cuda"
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    from dinov2.data.datasets.s2_csv import S2CsvDataset

    norm_stats = PRECOMPUTED_STATS

    train_ds = S2CsvDataset(
        csv_path=args.train_csv,
        ds_cfg_name="s2_12band",
        normalize_stats=norm_stats,
        scale_to_unit=True,
        pad_to_multiple=args.pad_to_multiple,
        compute_stats=False,
    )

    test_ds = S2CsvDataset(
        csv_path=args.test_csv,
        ds_cfg_name="s2_12band",
        normalize_stats=norm_stats,
        scale_to_unit=True,
        pad_to_multiple=args.pad_to_multiple,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    print(f"Using device={device}, train_samples={len(train_ds)}, test_samples={len(test_ds)}", flush=True)

    backbone = load_backbone(args.weights, device=device, debug=args.debug).to(device)
    if args.debug and device.type == "cuda":
        torch.cuda.synchronize()
        print(
            f"Backbone on {next(backbone.parameters()).device}, "
            f"cuda_allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MiB",
            flush=True,
        )
    head = MLPHead(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": backbone.parameters(), "lr": args.backbone_lr},
            {"params": head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    wandb_run = init_wandb(args)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch} started", flush=True)
        backbone.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for step, (x_dict, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            x_dict = to_device_batch(x_dict, device)
            feats = backbone(x_dict)
            logits = head(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                        {"train_loss_running": running_loss, "train_acc_running": running_acc, "step": step, "epoch": epoch}
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
                x_dict = to_device_batch(x_dict, device)
                feats = backbone(x_dict)
                logits = head(feats)
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
            f"recall={recall:.4f} fpr={fpr:.4f} auroc={test_auroc:.4f}"
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
    parser = argparse.ArgumentParser(description="Finetune Panopticon backbone plus 3-layer MLP head.")
    parser.add_argument("--train_csv", default="data_csv/96_3_train.csv")
    parser.add_argument("--test_csv", default="data_csv/96_3_test.csv")
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the backbone during finetuning (typically smaller than head lr).",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--device", default="cpu", help='PyTorch device string, e.g. "cpu" or "cuda".')
    parser.add_argument(
        "--stats_subset",
        type=float,
        default=None,
        help="If set, limit samples for mean/std (float fraction or int if whole number).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
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
    args = parser.parse_args()
    main(args)
