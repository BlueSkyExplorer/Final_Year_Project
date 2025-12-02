"""Utility helpers for validating dataloaders and datasets."""


def ensure_dataset_not_empty(loader, loader_name: str) -> None:
    """Raise a clear error when a dataloader has no underlying data.

    Args:
        loader: A ``torch.utils.data.DataLoader`` instance whose dataset will be checked.
        loader_name: Human-readable name of the loader used for error messages.

    Raises:
        ValueError: If ``len(loader.dataset)`` equals zero.
    """

    if len(loader.dataset) == 0:
        raise ValueError(
            f"{loader_name} dataset is empty. Please verify your data preparation and split configuration."
        )
