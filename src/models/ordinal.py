import torch
import torch.nn.functional as F


def coral_logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return torch.sum(probs > 0.5, dim=1)


def corn_logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    cum_probs = torch.cumprod(torch.sigmoid(logits), dim=1)
    thresholds = (cum_probs > 0.5).sum(dim=1)
    return thresholds


def coral_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    targets = targets.view(-1, 1)
    expanded = targets >= torch.arange(num_classes - 1, device=targets.device).unsqueeze(0)
    return F.binary_cross_entropy_with_logits(logits, expanded.float())


def corn_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    targets = targets.view(-1)
    loss = 0.0
    for k in range(num_classes - 1):
        mask = targets >= k
        if mask.any():
            loss += F.binary_cross_entropy_with_logits(logits[mask, k], torch.ones_like(logits[mask, k]))
        mask = targets < k
        if mask.any():
            loss += F.binary_cross_entropy_with_logits(logits[mask, k], torch.zeros_like(logits[mask, k]))
    return loss / (num_classes - 1)


def _ordinal_logits_to_class_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert ordinal threshold logits [B, K-1] to class probabilities [B, K]."""
    threshold_probs = torch.sigmoid(logits)
    class_probs = torch.zeros(logits.size(0), num_classes, device=logits.device, dtype=logits.dtype)
    class_probs[:, 0] = 1.0 - threshold_probs[:, 0]
    for cls_idx in range(1, num_classes - 1):
        class_probs[:, cls_idx] = threshold_probs[:, cls_idx - 1] - threshold_probs[:, cls_idx]
    class_probs[:, num_classes - 1] = threshold_probs[:, num_classes - 2]
    # Numerical guard for imperfect monotonicity.
    class_probs = class_probs.clamp_min(1e-6)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
    return class_probs


def corn_logits_to_class_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert CORN logits [B, K-1] to class probabilities [B, K]."""
    cum_probs = torch.cumprod(torch.sigmoid(logits), dim=1)
    class_probs = torch.zeros(logits.size(0), num_classes, device=logits.device, dtype=logits.dtype)
    class_probs[:, 0] = 1.0 - cum_probs[:, 0]
    for cls_idx in range(1, num_classes - 1):
        class_probs[:, cls_idx] = cum_probs[:, cls_idx - 1] - cum_probs[:, cls_idx]
    class_probs[:, num_classes - 1] = cum_probs[:, num_classes - 2]
    class_probs = class_probs.clamp_min(1e-6)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
    return class_probs


def distance_aware_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    alpha: float = 1.0,
) -> torch.Tensor:
    if logits.size(1) == num_classes:
        probs = F.softmax(logits, dim=1)
    elif logits.size(1) == num_classes - 1:
        probs = _ordinal_logits_to_class_probs(logits, num_classes)
    else:
        raise ValueError(
            f"Expected logits with shape [B, {num_classes}] or [B, {num_classes - 1}], got {tuple(logits.shape)}"
        )

    class_indices = torch.arange(num_classes, device=logits.device).unsqueeze(0)
    distance_weights = torch.abs(class_indices - targets.unsqueeze(1).long()).float().pow(alpha)
    wrong_class_mask = 1.0 - F.one_hot(targets.long(), num_classes=num_classes).float()
    penalty = -torch.log1p(-(probs.clamp(max=1 - 1e-7)))
    return torch.mean(torch.sum(distance_weights * wrong_class_mask * penalty, dim=1))
