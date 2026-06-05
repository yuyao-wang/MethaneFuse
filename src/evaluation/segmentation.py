"""Segmentation losses and metrics for MethaneFuse."""

from __future__ import annotations

import torch

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def iou_plus_scores_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> torch.Tensor:
    pred = torch.sigmoid(logits) >= float(threshold)
    gt = targets >= 0.5
    pred = pred.flatten(start_dim=1)
    gt = gt.flatten(start_dim=1)

    inter = torch.logical_and(pred, gt).sum(dim=1).float()
    union = torch.logical_or(pred, gt).sum(dim=1).float().clamp_min(1.0)
    iou = inter / union

    gt_empty = ~gt.any(dim=1)
    pred_empty = ~pred.any(dim=1)
    ones = torch.ones_like(iou)
    zeros = torch.zeros_like(iou)
    return torch.where(gt_empty, torch.where(pred_empty, ones, zeros), iou)
