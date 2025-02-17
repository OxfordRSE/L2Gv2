from .registry import get_dataset, register_dataset, DATASET_REGISTRY
from .base import GraphDataLoader
from .as733 import AS733Dataset
from .cora import planetoid_cora

def list_available_datasets():
    """
    Returns a list of registered dataset names.
    """
    return list(DATASET_REGISTRY.keys())

__all__ = ["get_dataset", "register_dataset", "DATASET_REGISTRY", "GraphDataLoader", "AS733Dataset", "planetoid_cora", "list_available_datasets"]