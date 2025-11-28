import torch
import torch.nn.functional as F
from src.models import ordinal as ordinal_utils


def coral_loss(logits, targets, num_classes: int = 4):
    return ordinal_utils.coral_loss(logits, targets, num_classes)


def corn_loss(logits, targets, num_classes: int = 4):
    return ordinal_utils.corn_loss(logits, targets, num_classes)


def distance_aware_loss(logits, targets, num_classes: int = 4):
    # logits expected to be of shape [B, num_classes]
    return ordinal_utils.distance_aware_loss(logits, targets, num_classes)
