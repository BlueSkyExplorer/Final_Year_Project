import torch
import torch.nn.functional as F
from typing import Optional


def cross_entropy_loss(logits, targets, class_weights: Optional[torch.Tensor] = None):
    return F.cross_entropy(logits, targets, weight=class_weights)


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute class-balanced weights using effective number of samples.

    Based on Cui et al., "Class-Balanced Loss Based on Effective Number
    of Samples", CVPR 2019.

    Effective number for class i:  E_i = (1 - beta^{n_i}) / (1 - beta)
    Weight for class i:            w_i = 1 / E_i = (1 - beta) / (1 - beta^{n_i})

    The weights are then normalised so that they sum to ``num_classes``
    (i.e. the average weight equals 1), preserving the same loss scale as
    unweighted cross-entropy.

    Args:
        targets: 1-D tensor of integer class labels.
        num_classes: total number of classes.
        beta: effective number hyper-parameter (typically 0.9, 0.99, 0.999,
              or 0.9999).  Must satisfy 0 <= beta < 1.
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    counts = torch.bincount(targets, minlength=num_classes).float()
    # Effective number: E_i = (1 - beta^n_i) / (1 - beta)
    # Weight:           w_i = (1 - beta) / (1 - beta^n_i)
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    # Normalise so that sum(weights) == num_classes
    weights = weights / weights.sum() * num_classes
    return weights


def class_balanced_ce(
    logits,
    targets,
    class_weights: Optional[torch.Tensor] = None,
    beta: float = 0.9999,
):
    """Cross-entropy weighted by effective-number class-balanced weights."""
    if class_weights is None:
        class_weights = compute_class_weights(
            targets, num_classes=logits.size(1), beta=beta,
        )
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


def cdw_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Class Distance Weighted Cross-Entropy (CDW-CE) loss.

    Based on Polat et al., "Class Distance Weighted Cross-Entropy Loss
    for Ulcerative Colitis Severity Estimation", 2022.

    CDW-CE = - sum_{i != c} log(1 - y_hat_i) * |i - c|^alpha

    where c is the ground-truth class index, y_hat_i is the predicted
    softmax probability for class i, and alpha controls the strength
    of the distance penalty.

    This loss operates on **multiclass softmax outputs** (N classes)
    as described in the original paper.

    Args:
        logits: raw logits of shape [B, num_classes].
        targets: integer class labels of shape [B].
        num_classes: total number of classes (default 4 for MES 0-3).
        alpha: distance penalty exponent (paper optimal ~5-6 for LIMUC).
    """
    probs = F.softmax(logits, dim=1)

    class_indices = torch.arange(num_classes, device=logits.device).unsqueeze(0)  # [1, K]
    targets_col = targets.unsqueeze(1).long()  # [B, 1]

    # Distance weight: |i - c|^alpha;  equals 0 when i == c
    distance_weights = torch.abs(class_indices - targets_col).float().pow(alpha)  # [B, K]

    # -log(1 - y_hat_i):  penalty for non-zero probability on wrong classes
    penalty = -torch.log1p(-(probs.clamp(max=1.0 - 1e-7)))  # [B, K]

    # distance_weights already zeros out the true-class column (|c-c|^a = 0),
    # so no explicit mask is needed — matching the paper formulation.
    return torch.mean(torch.sum(distance_weights * penalty, dim=1))
