"""Evaluation metrics shared by training and evaluation entry points."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _binary_auroc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC for binary labels {0,1} using rank statistics (tie-aware)."""
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape[0] != scores.shape[0]:
        return float("nan")
    n = labels.shape[0]
    if n == 0:
        return float("nan")
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[pos_mask].sum())
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray, pos_scores: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    preds = preds.astype(np.int64, copy=False)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "fpr": _safe_ratio(fp, fp + tn),
        "recall": _safe_ratio(tp, tp + fn),
        "auroc": _binary_auroc_from_scores(labels, pos_scores),
    }


def compute_split_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    pos_scores: Sequence[float],
) -> Dict[str, float]:
    count = int(len(labels))
    out: Dict[str, float] = {
        "count": float(count),
        "acc": float("nan"),
        "fpr": float("nan"),
        "recall": float("nan"),
        "auroc": float("nan"),
    }
    if count == 0 or len(preds) != count:
        return out
    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    out["acc"] = float((labels_np == preds_np).mean())
    if len(pos_scores) == count:
        out.update(
            compute_binary_metrics(
                labels=labels_np,
                preds=preds_np,
                pos_scores=np.asarray(pos_scores, dtype=np.float64),
            )
        )
    return out


def mean_logits_by_row(
    logits: torch.Tensor,
    sample_rows: torch.Tensor,
    *,
    num_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape={tuple(logits.shape)}")
    if sample_rows.ndim != 1 or sample_rows.shape[0] != logits.shape[0]:
        raise ValueError(
            "sample_rows must be rank-1 and match logits length: "
            f"sample_rows={tuple(sample_rows.shape)}, logits={tuple(logits.shape)}"
        )
    logits_f32 = logits.float()
    num_classes = logits_f32.shape[-1]
    row_sums = torch.zeros((num_rows, num_classes), device=logits.device, dtype=torch.float32)
    row_counts = torch.zeros((num_rows, 1), device=logits.device, dtype=torch.float32)
    row_sums.index_add_(0, sample_rows, logits_f32)
    ones = torch.ones((sample_rows.shape[0], 1), device=logits.device, dtype=torch.float32)
    row_counts.index_add_(0, sample_rows, ones)
    valid_mask = row_counts.squeeze(1) > 0
    row_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if row_ids.numel() == 0:
        return row_ids, row_sums.new_zeros((0, num_classes))
    row_logits = row_sums.index_select(0, row_ids) / row_counts.index_select(0, row_ids).clamp_min(1.0)
    return row_ids, row_logits
