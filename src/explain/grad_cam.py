import torch
import torch.nn.functional as F
from torch import nn
import cv2
import numpy as np


def get_last_conv_layer(model: nn.Module):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise RuntimeError("No convolutional layer found for Grad-CAM")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or get_last_conv_layer(model)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_category=None):
        output = self.model(input_tensor)
        if target_category is None:
            target_category = output.argmax(dim=1)
        loss = output[range(output.size(0)), target_category]
        self.model.zero_grad()
        loss.backward(torch.ones_like(loss))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze(1)


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay / overlay.max(), 0, 1)
    return overlay
