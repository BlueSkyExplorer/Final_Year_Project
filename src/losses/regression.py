import torch
import torch.nn.functional as F


def mse_loss(preds, targets):
    return F.mse_loss(preds, targets.float())


def huber_loss(preds, targets, delta: float = 1.0):
    return F.huber_loss(preds, targets.float(), delta=delta)


def regression_to_class(preds):
    return torch.clamp((preds >= 0.5).long() + (preds >= 1.5).long() + (preds >= 2.5).long(), max=3)
