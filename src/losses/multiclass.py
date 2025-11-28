import torch
import torch.nn.functional as F
from typing import Optional


def cross_entropy_loss(logits, targets, class_weights: Optional[torch.Tensor] = None):
    return F.cross_entropy(logits, targets, weight=class_weights)


def class_balanced_ce(logits, targets):
    classes = torch.unique(targets)
    counts = torch.tensor([torch.sum(targets == c) for c in classes], device=targets.device, dtype=torch.float)
    weights = 1.0 / (counts + 1e-6)
    weight_tensor = torch.zeros(logits.size(1), device=targets.device)
    for cls, w in zip(classes, weights):
        weight_tensor[cls] = w
    weight_tensor = weight_tensor / weight_tensor.sum() * len(classes)
    return F.cross_entropy(logits, targets, weight=weight_tensor)


def focal_loss(logits, targets, gamma: float = 2.0, alpha: float = 0.25):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()
