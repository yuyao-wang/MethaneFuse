import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Make the repository root importable when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CDIR", str(REPO_ROOT / "thirdparty" / "dinov2" / "configs"))

# Disable xFormers kernels to avoid long CUDA discovery/initialization hangs.
os.environ.setdefault("XFORMERS_DISABLED", "1")

PRECOMPUTED_STATS = (
    [786.128173828125, 1025.8876953125, 1593.730712890625, 2315.26123046875, 2710.462890625, 3115.90087890625, 3289.0830078125, 3465.536376953125, 3495.579833984375, 3517.7958984375, 4180.28564453125, 3567.866943359375],
    [435.72607421875, 597.6113891601562, 688.5059814453125, 840.1614990234375, 801.7208251953125, 706.9466552734375, 689.823974609375, 727.5567626953125, 668.30224609375, 551.3565063476562, 629.679931640625, 641.590087890625],
)

from thirdparty.dinov2.data.datasets.s2_csv import S2TemporalCsvDataset


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
            print(f"[Cache] Warmup complete. Cached: {len(unique_paths)-fallback_count}, Remote: {fallback_count}")


class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            path = self._local_file_cache.ensure_local(path)
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)


class CLSHead(nn.Module):
    """Minimal DINO-style classifier head: CLS -> LayerNorm -> Linear."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls))


def parse_band_indices(value: Optional[str]) -> Tuple[int, ...]:
    """Parse a comma-separated band-index list, validated against the 12 S2 bands."""

    if not value:
        return ()
    indices = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if not 0 <= index < 12:
            raise ValueError(f"band index {index} outside the 12 S2 bands")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate band indices in {value!r}")
    return tuple(indices)


class ConcatTemporalDataset(Dataset):
    """
    Wraps S2TemporalCsvDataset to concatenate three 12-band timepoints into one 36-band tensor.
    """

    def __init__(
        self,
        base_ds,
        legacy_t0_cdse_contract: bool = False,
        temporal_mode: str = "three",
        zero_t0_band_indices: Sequence[int] = (),
    ):
        self.base_ds = base_ds
        self.legacy_t0_cdse_contract = legacy_t0_cdse_contract
        self.temporal_mode = temporal_mode
        self.zero_t0_band_indices = tuple(int(index) for index in zero_t0_band_indices)
        if self.zero_t0_band_indices and self.legacy_t0_cdse_contract:
            raise ValueError(
                "zero_t0_band_indices and legacy_t0_cdse_contract both rewrite t0; "
                "enable only one so the ablation stays interpretable."
            )
        self.mean = torch.tensor(PRECOMPUTED_STATS[0], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(PRECOMPUTED_STATS[1], dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_list, label = self.base_ds[idx]  # list of 3 dicts
        if self.legacy_t0_cdse_contract:
            t0 = x_list[0]["imgs"]
            source_t0 = t0 * self.std + self.mean
            raw_t0 = torch.zeros_like(source_t0)
            retained_channels = [0, 1, 2, 3, 4, 5, 6, 10, 11]
            retained = source_t0[retained_channels]
            raw_t0[retained_channels] = torch.where(
                retained.abs() < 0.5,
                torch.zeros_like(retained),
                retained + 1000,
            )
            legacy_b8a = source_t0[8]
            raw_t0[7] = torch.where(
                legacy_b8a.abs() < 0.5,
                torch.zeros_like(legacy_b8a),
                legacy_b8a + 1000,
            )
            x_list[0]["imgs"] = (raw_t0 - self.mean) / self.std
        if self.zero_t0_band_indices:
            # Reproduce the legacy t0 band contract: those raw DN planes are
            # identically zero at t0 only, while t90/t360 keep real values.
            # Zeroing happens in raw DN space, then the tensor is renormalized,
            # so the model sees exactly what the legacy loader produced.
            t0 = x_list[0]["imgs"]
            raw_t0 = t0 * self.std + self.mean
            for band_index in self.zero_t0_band_indices:
                raw_t0[band_index] = 0.0
            x_list[0]["imgs"] = (raw_t0 - self.mean) / self.std
        if self.temporal_mode == "three":
            selected = x_list
        elif self.temporal_mode == "history":
            # Forecasting view: the current visit is withheld entirely and only
            # the earlier visits are shown. Panopticon consumes a variable
            # channel count via chn_ids, so dropping t0 needs no zero padding
            # and leaks nothing from the target acquisition.
            selected = x_list[1:]
        elif self.temporal_mode == "history_oldest":
            # Site-identity control: a single earliest visit, no temporal pair.
            selected = x_list[2:3]
        else:
            selected = x_list[:1]
        imgs = torch.cat([x["imgs"] for x in selected], dim=0)
        chn_ids = torch.cat([x["chn_ids"] for x in selected], dim=0)
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

    from src.backbones import build_panopticon_vitb14

    if weights_path in (None, "", "none", "scratch", "random"):
        if debug:
            print("Building backbone from scratch (no pretrained weights)", flush=True)
        return build_panopticon_vitb14()

    weights_path = Path(weights_path)
    # if not weights_path.is_file():
    #     alt_path = Path(str(weights_path) + "?download=true")
    #     if alt_path.is_file():
    #         weights_path = alt_path
    #     else:
    #         raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading checkpoint from {weights_path}", flush=True)
    if debug:
        print("Building backbone...", flush=True)
    model = build_panopticon_vitb14()
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

def resize_imgs(x_dict, output_size: int):
    """Resize a batch before the backbone."""

    imgs = x_dict.get("imgs")
    if imgs is None:
        return x_dict

    # Support both (C, H, W) and (B, C, H, W) shapes.
    squeeze_back = False
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
        squeeze_back = True

    if imgs.ndim == 4:
        imgs = F.interpolate(
            imgs,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
        if squeeze_back:
            imgs = imgs.squeeze(0)
        x_dict["imgs"] = imgs

    return x_dict


def apply_spatial_contract(x_dict, downsample_size: int, output_size: int):
    """Apply an explicit spatial bottleneck before the model input resize."""

    if downsample_size > 0:
        x_dict = resize_imgs(x_dict, downsample_size)
    if output_size > 0:
        x_dict = resize_imgs(x_dict, output_size)
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


def default_run_name(args) -> str:
    """Build a repeatable identifier from datasets and whether backbone trains."""
    train_stem = Path(args.train_csv).stem or "train"
    test_stem = Path(args.test_csv).stem or "test"
    mode = "ft" if args.train_backbone else "head"
    return f"{train_stem}__{test_stem}__{mode}"


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
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
            "best_train_acc": best_train_acc,
            "best_test_acc": best_test_acc,
            "args": vars(args),
        },
        path,
    )


def try_resume(path: Path, backbone: nn.Module, head: nn.Module, optimizer, scheduler, device):
    if not path.is_file():
        return 1, 0, 0.0, 0.0
    ckpt = torch.load(path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_train_acc = ckpt.get("best_train_acc", 0.0)
    best_test_acc = ckpt.get("best_test_acc", 0.0)
    print(f"Resumed from {path} at epoch {start_epoch-1}", flush=True)
    return start_epoch, global_step, best_train_acc, best_test_acc


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "noam":
        warmup_steps = max(args.warmup_steps, 1)
        noam_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def main(args):
    seeded = args.seed >= 0
    if seeded:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[Reproducibility] seed={args.seed}", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    norm_stats = PRECOMPUTED_STATS
    ds_kwargs = {
        "ds_cfg_name": "s2_12band",
        "normalize_stats": norm_stats,
        "scale_to_unit": False,
        "pad_to_multiple": args.pad_to_multiple,
        "compute_stats": False,
        "path_columns": (args.t0_col, args.t90_col, args.t360_col),
    }

    dataset_cls = S2TemporalCsvDataset
    cache_obj = None
    if args.local_cache_dir:
        dataset_cls = CachedS2TemporalCsvDataset
        cache_obj = StaticAnchoredCache(args.local_cache_dir)
        ds_kwargs["local_file_cache"] = cache_obj

    base_train_ds = dataset_cls(
        csv_path=args.train_csv,
        **ds_kwargs,
    )
    base_test_ds = dataset_cls(
        csv_path=args.test_csv,
        **ds_kwargs,
    )

    if args.local_cache_warmup and cache_obj:
        all_paths = []
        for ds in (base_train_ds, base_test_ds):
            for col in ds.path_columns:
                all_paths.extend(ds.df[col].tolist())
        cache_obj.warm_up(all_paths, max_workers=args.local_cache_workers)

    zero_t0_band_indices = parse_band_indices(args.zero_t0_band_indices)
    train_ds = ConcatTemporalDataset(
        base_train_ds,
        legacy_t0_cdse_contract=args.legacy_t0_cdse_contract,
        temporal_mode=args.temporal_mode,
        zero_t0_band_indices=zero_t0_band_indices,
    )
    test_ds = ConcatTemporalDataset(
        base_test_ds,
        legacy_t0_cdse_contract=args.legacy_t0_cdse_contract,
        temporal_mode=args.temporal_mode,
        zero_t0_band_indices=zero_t0_band_indices,
    )
    if zero_t0_band_indices:
        print(
            "[Data] t0 band ablation: raw DN set to zero for band indices "
            f"{list(zero_t0_band_indices)} at t0 only; t90/t360 untouched.",
            flush=True,
        )
    if args.legacy_t0_cdse_contract:
        print(
            "[Data] Applying legacy CDSE t0 contract: +1000 DN; B8A moved into "
            "the B8 slot; B8A/B9 slots set to zero.",
            flush=True,
        )
    print(f"[Data] Temporal mode: {args.temporal_mode}", flush=True)
    if args.contract_downsample_size > 0:
        print(
            "[Data] Spatial contract: explicit "
            f"{args.contract_downsample_size}x{args.contract_downsample_size} "
            f"bottleneck -> {args.input_resize_size}x{args.input_resize_size}, "
            "bilinear align_corners=False.",
            flush=True,
        )

    pin_memory = device.type == "cuda"
    train_generator = None
    if seeded:
        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
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
    best_test_path = ckpt_dir / "ckpt_best_test.pth"

    start_epoch = 1
    global_step = 0
    best_train_acc = 0.0
    best_test_acc = 0.0
    if args.resume:
        start_epoch, global_step, best_train_acc, best_test_acc = try_resume(
            latest_path, backbone, head, optimizer, scheduler, device
        )

    wandb_run = init_wandb(args)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_generator is not None:
            # The convergence queue invokes one target epoch per process. Seeding
            # by epoch keeps shuffle order reproducible without replaying the same
            # order after every resume.
            train_generator.manual_seed(args.seed + epoch)
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
            x_dict = apply_spatial_contract(
                x_dict,
                args.contract_downsample_size,
                args.input_resize_size,
            )
            

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
                x_dict = apply_spatial_contract(
                    x_dict,
                    args.contract_downsample_size,
                    args.input_resize_size,
                )
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
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        test_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
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
            f"f1={test_f1:.4f} recall={recall:.4f} fpr={fpr:.4f} "
            f"auroc={test_auroc:.4f}",
            flush=True,
        )
        if args.metrics_jsonl:
            metrics_path = Path(args.metrics_jsonl)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                            "test_f1": test_f1,
                            "test_recall": recall,
                            "test_fpr": fpr,
                            "test_auroc": test_auroc,
                            "tp": tp,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                        }
                    )
                    + "\n"
                )

        # --- save checkpoints ---
        prev_best_train = best_train_acc
        prev_best_test = best_test_acc
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)

        if args.save_checkpoints:
            save_checkpoint(
                latest_path,
                epoch,
                global_step,
                backbone,
                head,
                optimizer,
                scheduler,
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
                    best_train_acc,
                    best_test_acc,
                    args,
                )

            if test_acc > prev_best_test:
                save_checkpoint(
                    best_test_path,
                    epoch,
                    global_step,
                    backbone,
                    head,
                    optimizer,
                    scheduler,
                    best_train_acc,
                    best_test_acc,
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
                    "test_f1": test_f1,
                    "test_recall": recall,
                    "test_fpr": fpr,
                    "test_auroc": test_auroc,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panopticon ViT + CLS head with concatenated temporal S2 inputs.")
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--test_csv", default="")
    parser.add_argument(
        "--weights",
        default="weights/panopticon_vitb14_teacher.pth",
        help='Checkpoint path. Use "none" to train the ViT backbone from scratch.',
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Non-negative seed enables paired, epoch-stable data/model randomness.",
    )
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
        "--input_resize_size",
        type=int,
        default=0,
        help="Resize inputs to this square size after normalization; <=0 disables.",
    )
    parser.add_argument(
        "--zero_t0_band_indices",
        type=str,
        default="",
        help=(
            "Comma-separated band indices whose raw DN is set to zero at t0 only "
            "(t90/t360 untouched). Use 8,9 to give pure-GEE data the legacy t0 "
            "band contract, which has B8A/B9 identically zero at t0. Mutually "
            "exclusive with --legacy_t0_cdse_contract, which also adds +1000 DN."
        ),
    )
    parser.add_argument(
        "--contract_downsample_size",
        type=int,
        default=0,
        help=(
            "Optional explicit spatial bottleneck before input_resize_size. "
            "Use 36 to compare stored legacy 224 imagery and native-36 imagery "
            "under the same 36->224 runtime sampling contract."
        ),
    )
    parser.add_argument(
        "--legacy_t0_cdse_contract",
        action="store_true",
        help="Transform pure-GEE t0 to the radiometric/channel contract used by the old CDSE t0 data.",
    )
    parser.add_argument(
        "--temporal_mode",
        choices=["three", "t0", "history", "history_oldest"],
        default="three",
        help=(
            "three: all visits. t0: current visit only. history: earlier visits "
            "only, withholding t0 (forecasting view). history_oldest: the single "
            "earliest visit only, a site-identity control for `history`."
        ),
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
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="panopticon", help="WandB project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional WandB run name.")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Base directory to store checkpoints (latest/best).",
    )
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run name for checkpoint subfolder. Defaults to <train_csv_stem>__<test_csv_stem>__ft|head.",
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
    parser.add_argument("--metrics_jsonl", default="", help="Optional per-epoch metrics JSONL path.")
    args = parser.parse_args()
    main(args)
