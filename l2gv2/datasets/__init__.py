from .registry import get_dataset, register_dataset, DATASET_REGISTRY
from .base import BaseDataset
from .as733 import AS733Dataset
from .cora import CoraDataset


def list_available_datasets():
    """
    Returns a list of registered dataset names.
    """
    return list(DATASET_REGISTRY.keys())


__all__ = [
    "get_dataset",
    "register_dataset",
    "DATASET_REGISTRY",
    "BaseDataset",
    "AS733Dataset",
    "CoraDataset",
    "list_available_datasets",
]
