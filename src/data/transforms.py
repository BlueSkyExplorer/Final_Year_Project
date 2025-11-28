from torchvision import transforms


def build_transforms(cfg, train: bool = True):
    size = cfg["images"].get("image_size", 224)
    norm = transforms.Normalize(mean=cfg["normalization"]["mean"], std=cfg["normalization"]["std"])
    if train:
        aug = [
            transforms.RandomResizedCrop(size, scale=tuple(cfg["augmentations"].get("random_resized_crop_scale", (0.8, 1.0)))),
            transforms.RandomHorizontalFlip(),
        ]
        if cfg["augmentations"].get("color_jitter", False):
            aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        aug.extend([transforms.ToTensor(), norm])
        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.Resize(size + 32),
            transforms.CenterCrop(cfg["augmentations"].get("center_crop", size)),
            transforms.ToTensor(),
            norm,
        ])
