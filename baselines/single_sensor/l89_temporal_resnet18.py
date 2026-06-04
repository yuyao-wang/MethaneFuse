import argparse
import hashlib
import os
import shutil
import sys
import random
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

# 路径导入逻辑
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dinov2.data.datasets.s2_csv import S2TemporalCsvDataset, _SkipSample
except ImportError:
    class _SkipSample(Exception): pass
    S2TemporalCsvDataset = object

os.environ.setdefault("XFORMERS_DISABLED", "1")

# 默认统计值（作为回退方案）
DEFAULT_STATS = (
    [10000.0] * 21,
    [2000.0] * 21,
)

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
        if dst.exists(): return str(dst)
        if self._get_free_space() < self.min_free_bytes: return original
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, dst)
        except Exception:
            with suppress(FileNotFoundError): tmp.unlink()
            return original
        return str(dst)

    def warm_up(self, paths: Sequence[str], max_workers: int = 8) -> None:
        unique_paths = sorted({os.path.abspath(p) for p in paths if isinstance(p, str)})
        if not unique_paths: return
        print(f"[Cache] 预热中 (目标: {len(unique_paths)})...", flush=True)
        def _copy_one(path: str):
            res = self.ensure_local(path)
            return res == path
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, p) for p in unique_paths]
            fallback_count = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"[Cache] 预热完成。已缓存: {len(unique_paths)-fallback_count}, 远程: {fallback_count}")

class CachedS2TemporalCsvDataset(S2TemporalCsvDataset):
    def __init__(self, *args, local_file_cache: Optional[StaticAnchoredCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_file_cache = local_file_cache

    def _load_image(self, path: str, *, column_name=None, sample_id=None):
        if self._local_file_cache is not None and isinstance(path, str):
            path = self._local_file_cache.ensure_local(path)
        return super()._load_image(path, column_name=column_name, sample_id=sample_id)

class ConcatTemporalDataset(Dataset):
    def __init__(self, csv_path: str, **kwargs):
        local_cache_dir = kwargs.pop("local_cache_dir", None)
        cache_warmup = kwargs.pop("cache_warmup", False)
        cache_workers = kwargs.pop("cache_workers", 8)
        
        # 初始暂时不进行归一化（等计算完 Stats 后再手动更新）
        kwargs["normalize_stats"] = None 
        
        dataset_cls = S2TemporalCsvDataset
        self.cache_obj = None
        if local_cache_dir:
            dataset_cls = CachedS2TemporalCsvDataset
            self.cache_obj = StaticAnchoredCache(local_cache_dir)
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
        """动态更新归一化参数（只做一次 mean/std 标准化）"""
        std = torch.clamp(std, min=1e-6)
        self._mean_tensor = mean.view(-1, 1, 1)
        self._std_tensor = std.view(-1, 1, 1)
        # 仅记录，实际归一化在 __getitem__ 里完成
        # self._base.normalize_stats = (mean.tolist(), std.tolist())

    def __len__(self): return len(self._base)

    def __getitem__(self, idx):
        x_list, label = self._base[idx]
        imgs = torch.cat([x["imgs"] for x in x_list], dim=0) # (21, H, W)
        if self._mean_tensor is not None and self._std_tensor is not None:
            # 记录原始数据中为 0 (No-Data) 的位置
            mask = (imgs == 0)
            imgs = (imgs - self._mean_tensor) / self._std_tensor
            # 将 No-Data 区域恢复为 0，避免填充值变成负数干扰模型
            imgs[mask] = 0.0
        return imgs, int(label)

def compute_dynamic_stats(dataset: Dataset, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样并剔除 0 值（No-Data）后计算 21 通道的均值和标准差
    """
    print(f"[Stats] 正在随机采样 {n_samples} 个样本（剔除 0 值）计算 Mean/Std...", flush=True)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    # 用来存储每个通道的所有有效像素值
    # 注意：由于每个样本的有效像素数量不同，我们不能简单堆叠张量
    channel_sums = torch.zeros(21)
    channel_sq_sums = torch.zeros(21)
    channel_pixel_counts = torch.zeros(21)

    for i in tqdm(indices, desc="采样中"):
        try:
            img, _ = dataset[i] # img shape: (21, H, W)
            img = img.float()
            
            for c in range(21):
                channel_data = img[c]
                valid_pixels = channel_data[channel_data != 0] # 剔除 0 值
                
                if valid_pixels.numel() > 0:
                    channel_sums[c] += valid_pixels.sum()
                    channel_sq_sums[c] += (valid_pixels ** 2).sum()
                    channel_pixel_counts[c] += valid_pixels.numel()
                    
        except Exception as e:
            continue
            
    # 计算最终的均值和标准差
    # 使用公式: Var(X) = E[X^2] - (E[X])^2
    means = channel_sums / (channel_pixel_counts + 1e-6)
    variances = (channel_sq_sums / (channel_pixel_counts + 1e-6)) - (means ** 2)
    stds = torch.sqrt(torch.clamp(variances, min=1e-6))
    
    # 检查是否有通道完全没有有效数据
    if channel_pixel_counts.min() == 0:
        print("[Stats] 警告：某些通道未发现有效像素，已使用默认值填充")
        for c in range(21):
            if channel_pixel_counts[c] == 0:
                means[c], stds[c] = 10000.0, 2000.0

    print(f"[Stats] 计算完成（已剔除 0 值）。")
    print(f"Mean (first 3): {means[:3].tolist()}...")
    print(f"Std (first 3): {stds[:3].tolist()}...")
    
    return means, stds

def init_wandb(args):
    if not args.use_wandb: return None
    import wandb
    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

def build_resnet18(num_channels: int = 21, num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ... train_one_epoch 和 evaluate 函数保持不变 ...
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, wandb_run=None, scheduler=None):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if step % 50 == 0:
            if wandb_run: wandb_run.log({"train/loss_step": loss.item()})
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    tp = fp = fn = tn = 0
    all_probs, all_targets = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

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
    wandb_run = init_wandb(args)

    ds_kwargs = {
        "path_columns": (args.t0_col, args.t90_col, args.t360_col),
        "ds_cfg_name": "landsat89_7band",
        "pad_to_multiple": args.pad_to_multiple,
        "skip_invalid_samples": args.skip_invalid_samples,
        "local_cache_dir": args.local_cache_dir,
        "cache_workers": args.local_cache_workers,
    }

    # 1. 初始化训练集并预热缓存（如果启用）
    train_ds = ConcatTemporalDataset(csv_path=args.train_csv, cache_warmup=args.local_cache_warmup, **ds_kwargs)
    
    # 2. 动态计算统计值（此时数据已在本地缓存，读取速度快）
    mean, std = compute_dynamic_stats(train_ds, n_samples=args.stats_samples)
    train_ds.update_stats(mean, std)
    
    # 3. 初始化测试集并同步 Stats
    test_ds = ConcatTemporalDataset(csv_path=args.test_csv, cache_warmup=False, **ds_kwargs)
    test_ds.update_stats(mean, std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_resnet18(num_channels=21).to(device)
    if device.type == "cuda": model = nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 暂停使用 Noam 学习率调度
    scheduler = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, wandb_run, scheduler)
        te_loss, te_acc, te_recall, te_fpr, te_auroc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}: Train Acc={tr_acc:.4f} Train Loss={tr_loss:.4f} | "
            f"Test Acc={te_acc:.4f} Test Loss={te_loss:.4f} "
            f"Recall={te_recall:.4f} FPR={te_fpr:.4f} AUROC={te_auroc:.4f}"
        )
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/acc": tr_acc,
                    "train/loss": tr_loss,
                    "test/acc": te_acc,
                    "test/loss": te_loss,
                    "test/recall": te_recall,
                    "test/fpr": te_fpr,
                    "test/auroc": te_auroc,
                    "stats/mean": mean.tolist(),
                }
            )

    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... 原有参数 ...
    parser.add_argument("--train_csv", default="data_csv/train.csv")
    parser.add_argument("--test_csv", default="data_csv/test.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_warmup", action="store_true")
    parser.add_argument("--local_cache_workers", type=int, default=12)
    parser.add_argument("--stats_samples", type=int, default=1000, help="用于计算统计值的采样样本数")
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--skip_invalid_samples", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="baselines")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--t0_col", default="path_t0")
    parser.add_argument("--t90_col", default="path_t90")
    parser.add_argument("--t360_col", default="path_t360")
    args = parser.parse_args()
    main(args)
