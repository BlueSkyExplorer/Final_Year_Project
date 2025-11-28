import torch.nn as nn
import torchvision.models as models


BACKBONE_OUT_FEATURES = {
    "resnet18": 512,
    "efficientnet_b0": 1280,
}


def build_backbone(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone {name}")
    return model


def get_backbone_output_dim(name: str) -> int:
    if name.lower() not in BACKBONE_OUT_FEATURES:
        raise ValueError(f"Unknown backbone {name}")
    return BACKBONE_OUT_FEATURES[name.lower()]
