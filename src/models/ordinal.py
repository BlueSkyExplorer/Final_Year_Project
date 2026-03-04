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
    count = 0
    for k in range(num_classes - 1):
        # CORN conditional: 只考慮 Y ≥ k 的樣本
        mask = targets >= k
        if mask.sum() <= 0:
            continue
        # 在 Y ≥ k 的樣本中: Y > k → 正 (1), Y = k → 負 (0)
        conditional_labels = (targets[mask] > k).float()
        loss += F.binary_cross_entropy_with_logits(
            logits[mask, k], conditional_labels
        )
        count += 1
    return loss / max(count, 1)


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


