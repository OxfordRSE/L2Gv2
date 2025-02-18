from torch_geometric.datasets import Planetoid
from pathlib import Path
from .registry import register_dataset


@register_dataset("Cora")
def planetoid_cora(root: str | None = None, **kwargs):
    """
    Cora dataset from the Planetoid dataset collection.
    """
    kwargs.setdefault("name", "Cora")
    if root is None:
        root = str(Path(__file__).parent.parent.parent / "data" / "cora")
    return Planetoid(root, **kwargs)
