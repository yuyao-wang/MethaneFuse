import argparse
import os
from pathlib import Path
import sys
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

PRECOMPUTED_STATS = (
    [786.128173828125, 1025.8876953125, 1593.730712890625, 2315.26123046875, 2710.462890625, 3115.90087890625, 3289.0830078125, 3465.536376953125, 3495.579833984375, 3517.7958984375, 4180.28564453125, 3567.866943359375],
    [435.72607421875, 597.6113891601562, 688.5059814453125, 840.1614990234375, 801.7208251953125, 706.9466552734375, 689.823974609375, 727.5567626953125, 668.30224609375, 551.3565063476562, 629.679931640625, 641.590087890625],
    # [1818.0352783203125, 1985.090576171875, 2360.004638671875, 2737.189453125, 3120.86328125, 3449.22607421875, 3621.799072265625, 3876.05810546875, 0.0, 0.0, 4580.51171875, 3736.6298828125],
    # [322.40814208984375, 444.110595703125, 546.5563354492188, 702.7158203125, 717.8546142578125, 661.7835693359375, 648.707763671875, 623.9007568359375, 9.999999974752427e-07, 9.999999974752427e-07, 665.3115844726562, 724.7693481445312],
    # -7 case
    # [1807.9029541015625, 1980.2451171875, 2359.39892578125, 2759.558837890625, 3152.531982421875, 3500.52587890625, 3681.44287109375, 3925.999755859375, 0.0, 0.0, 4612.71875, 3814.356689453125],
    # [309.2794494628906, 425.18365478515625, 522.7911376953125, 699.1823120117188, 718.7879638671875, 669.5192260742188, 667.4446411132812, 653.8464965820312, 9.999999974752427e-07, 9.999999974752427e-07, 667.0552978515625, 720.5349731445312],
    # train+test
    # [2023.93408203125, 2267.178466796875, 2703.1396484375, 3257.457275390625, 3589.15478515625, 3874.7529296875, 4058.203125, 4229.97802734375, 0.0, 0.0, 4915.81591796875, 4429.00439453125],
    # [659.8005981445312, 759.5709838867188, 878.8653564453125, 1164.0340576171875, 1186.4278564453125, 1077.7305908203125, 1084.9752197265625, 1064.8203125, 9.999999974752427e-07, 9.999999974752427e-07, 1363.797119140625, 1409.4805908203125],
)


class MLPHead(nn.Module):
    """Three-layer MLP head for binary classification."""

    def __init__(self, in_dim: int = 768, hidden: Tuple[int, int] = (1024, 512), num_classes: int = 2, drop: float = 0.3):
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


class TemporalFusion(nn.Module):
    """
    Late fusion over three timepoints.
    - diff: concat pairwise differences [f(t0)-f(t-90), f(t0)-f(t-360)]
    - attn: cross-attend h0 over [h90, h360], output concat [h0, delta]
    """

    def __init__(self, embed_dim: int, mode: str = "diff", num_layers: int = 1, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if mode == "concat":
            # Backward compatibility: treat concat as diff mode.
            mode = "diff"
        self.mode = mode
        if mode == "diff":
            self.out_dim = embed_dim * 2
            self.time_pos = None
            self.encoder = None
            self.norm = nn.LayerNorm(embed_dim * 2)
        elif mode == "attn":
            self.time_pos = None
            self.encoder = None
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.q_norm = nn.LayerNorm(embed_dim)
            self.kv_norm = nn.LayerNorm(embed_dim)
            self.delta_ff = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout),
            )
            self.out_norm = nn.LayerNorm(embed_dim * 2)
            self.out_dim = embed_dim * 2  # concat h0 and delta
        elif mode == "self_attn":
            self.time_pos = None
            self.encoder = None
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.role_embed_current = nn.Parameter(torch.zeros(embed_dim))
            self.role_embed_history = nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(self.role_embed_current, std=0.02)
            nn.init.normal_(self.role_embed_history, std=0.02)
            self.self_in_norm = nn.LayerNorm(embed_dim)
            self.self_ffn_norm = nn.LayerNorm(embed_dim)
            self.self_delta_ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.out_norm = nn.LayerNorm(embed_dim * 2)
            self.out_dim = embed_dim * 2  # concat h0 and delta
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, feats: Iterable[torch.Tensor]) -> torch.Tensor:
        feats = list(feats)
        if len(feats) != 3:
            raise ValueError(f"Expected 3 timepoints, got {len(feats)}")
        if self.mode == "diff":
            f0, f90, f360 = feats
            fused = torch.cat([f0 - f90, f0 - f360], dim=1)
            return self.norm(fused)
        if self.mode == "attn":
            h0, h90, h360 = feats
            q = self.q_norm(h0).unsqueeze(1)           # [B,1,D]
            kv = torch.stack([self.kv_norm(h90), self.kv_norm(h360)], dim=1)  # [B,2,D]
            delta, _ = self.attn(q, kv, kv)            # [B,1,D]
            delta = delta.squeeze(1)                   # [B,D]
            delta = delta + self.delta_ff(delta)       # residual MLP on delta
            return self.out_norm(torch.cat([h0, delta], dim=1))

        if self.mode == "self_attn":
            h0, h90, h360 = feats
            x = torch.stack([h0, h90, h360], dim=1)  # [B,3,D]
            x[:, 0, :] = x[:, 0, :] + self.role_embed_current
            x[:, 1:, :] = x[:, 1:, :] + self.role_embed_history

            x_norm = self.self_in_norm(x)
            z, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=True)  # [B,3,D]
            y = x + z

            y_norm = self.self_ffn_norm(y)
            y = y + self.self_delta_ff(y_norm)

            h0_prime = y[:, 0, :]
            delta = h0_prime - h0
            out = torch.cat([h0, delta], dim=1)
            return self.out_norm(out)

        raise ValueError(f"Unknown fusion mode during forward: {self.mode}")


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
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "num_workers": args.num_workers,
            "pad_to_multiple": args.pad_to_multiple,
            "device": args.device,
            "fusion": args.fusion,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "max_grad_norm": args.max_grad_norm,
        },
    )
    return run


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset

    norm_stats = PRECOMPUTED_STATS

    train_ds = S2TemporalCsvDataset(
        csv_path=args.train_csv,
        ds_cfg_name="s2_12band",
        normalize_stats=norm_stats,
        scale_to_unit=False,
        pad_to_multiple=args.pad_to_multiple,
        compute_stats=False,
        path_columns=(args.t0_col, args.t90_col, args.t360_col),
    )

    test_ds = S2TemporalCsvDataset(
        csv_path=args.test_csv,
        ds_cfg_name="s2_12band",
        normalize_stats=norm_stats,
        scale_to_unit=False,
        pad_to_multiple=args.pad_to_multiple,
        compute_stats=False,
        path_columns=(args.t0_col, args.t90_col, args.t360_col),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory
    )

    print(
        f"Using device={device}, fusion={args.fusion}, train_samples={len(train_ds)}, test_samples={len(test_ds)}",
        flush=True,
    )

    backbone = load_backbone(args.weights, device=device, debug=args.debug).to(device)
    fusion = TemporalFusion(
        embed_dim=args.embed_dim, mode=args.fusion, num_layers=args.attn_layers, num_heads=args.attn_heads, dropout=args.attn_dropout
    ).to(device)
    head_in_dim = fusion.out_dim
    head = MLPHead(in_dim=head_in_dim, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(
        [
            {"params": backbone.parameters(), "lr": args.backbone_lr},
            {"params": fusion.parameters(), "lr": args.lr},
            {"params": head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    # Noam-style scheduler: lr = base_lr * min(step^-0.5, step * warmup^-1.5)
    warmup_steps = max(args.warmup_steps, 1)
    noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    wandb_run = init_wandb(args)

    for epoch in range(1, args.epochs + 1):
        freeze_backbone = args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        print(f"Epoch {epoch} started (freeze_backbone={freeze_backbone})", flush=True)
        if freeze_backbone:
            set_trainable(backbone, False)
            backbone.eval()
        else:
            set_trainable(backbone, True)
            backbone.train()
        fusion.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for step, (x_list, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            x_list = [recursive_to_device(x, device) for x in x_list]
            feats = [backbone(x_dict) for x_dict in x_list]
            fused = fusion(feats)
            logits = head(fused)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                # Only clip trainable parameters.
                trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

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
        fusion.eval()
        backbone.eval()
        correct = 0
        total = 0
        test_loss_total = 0.0
        tp = fp = fn = tn = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for step, (x_list, labels) in enumerate(test_loader, 1):
                labels = labels.to(device)
                x_list = [recursive_to_device(x, device) for x in x_list]
                feats = [backbone(x_dict) for x_dict in x_list]
                fused = fusion(feats)
                logits = head(fused)
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
    parser = argparse.ArgumentParser(description="Temporal finetuning of Panopticon backbone plus MLP head.")
    parser.add_argument("--train_csv", default="data_csv/hongxuan_temporal_32/train.csv")
    parser.add_argument("--test_csv", default="data_csv/hongxuan_temporal_32/test.csv")
    parser.add_argument("--weights", default="weights/panopticon_vitb14_teacher.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the backbone during finetuning (typically smaller than head lr).",
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
    parser.add_argument(
        "--fusion",
        choices=["diff", "attn", "self_attn", "concat"],
        default="diff",
        help=(
            "Fusion mode for three timepoints: diff uses [f(t0)-f(t-90), f(t0)-f(t-360)] "
            "(concat alias to diff); attn uses cross-attention delta with output [h0, delta]; "
            "self_attn runs a 3-token self-attention over [h0,h90,h360] and returns [h0, h0'-h0]."
        ),
    )
    parser.add_argument("--embed_dim", type=int, default=768, help="Backbone output dimension (Panopticon teacher is 768).")
    parser.add_argument("--attn_layers", type=int, default=1, help="Number of Transformer encoder layers for attn fusion.")
    parser.add_argument("--attn_heads", type=int, default=4, help="Number of heads for attn fusion.")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Dropout for attn fusion.")
    parser.add_argument("--t0_col", default="image_path", help="CSV column for t0 image path.")
    parser.add_argument("--t90_col", default="s2_pre_path", help="CSV column for t-90 image path.")
    parser.add_argument("--t360_col", default="s2_pre_pre_path", help="CSV column for t-360 image path.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=10,
        help="Number of initial epochs to freeze the backbone (0 disables freezing).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm; set <=0 to disable.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Warmup steps for Noam LR scheduler.",
    )
    args = parser.parse_args()
    main(args)
