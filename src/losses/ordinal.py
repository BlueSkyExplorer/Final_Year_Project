import torch
import torch.nn.functional as F
from src.models import ordinal as ordinal_utils


def coral_loss(logits, targets, num_classes: int = 4):
    return ordinal_utils.coral_loss(logits, targets, num_classes)


def corn_loss(logits, targets, num_classes: int = 4):
    return ordinal_utils.corn_loss(logits, targets, num_classes)


def distance_aware_loss(logits, targets, num_classes: int = 4, distance_alpha: float = 1.0):
    # logits expected to be of shape [B, num_classes] or [B, num_classes - 1]
    return ordinal_utils.distance_aware_loss(logits, targets, num_classes, distance_alpha=distance_alpha)
