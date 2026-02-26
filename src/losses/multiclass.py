import torch
import torch.nn.functional as F
from typing import Optional


def cross_entropy_loss(logits, targets, class_weights: Optional[torch.Tensor] = None):
    return F.cross_entropy(logits, targets, weight=class_weights)


def compute_class_weights(targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = 1.0 / (counts + eps)
    return weights / weights.sum() * num_classes


def class_balanced_ce(logits, targets, class_weights: Optional[torch.Tensor] = None):
    if class_weights is None:
        class_weights = compute_class_weights(targets, num_classes=logits.size(1))
    return F.cross_entropy(logits, targets, weight=class_weights.to(logits.device))


def focal_loss(logits, targets, gamma: float = 2.0, alpha: float = 0.25):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()
