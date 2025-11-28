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


def distance_aware_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    distance = torch.abs(torch.arange(num_classes, device=logits.device).unsqueeze(0) - targets.unsqueeze(1))
    weighted = distance * target_one_hot
    return torch.sum(weighted * -torch.log(probs + 1e-6)) / logits.size(0)
