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


def focal_loss(
    logits,
    targets,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor | list[float] | float] = 0.25,
):
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    ce = -log_pt

    alpha_factor = torch.ones_like(pt)
    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_factor = alpha_factor * float(alpha)
        else:
            alpha_tensor = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=logits.dtype)
            alpha_tensor = alpha_tensor.to(device=logits.device, dtype=logits.dtype)
            if alpha_tensor.ndim == 0 or alpha_tensor.numel() == 1:
                alpha_factor = alpha_factor * alpha_tensor.reshape(())
            elif alpha_tensor.ndim == 1 and alpha_tensor.numel() == logits.size(1):
                alpha_factor = alpha_tensor.gather(0, targets)
            else:
                raise ValueError(
                    "alpha for focal_loss must be a scalar or a 1D tensor/list with length equal to num_classes"
                )

    loss = alpha_factor * (1 - pt) ** gamma * ce
    return loss.mean()
