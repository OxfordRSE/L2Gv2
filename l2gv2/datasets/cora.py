import logging
from torch_geometric.datasets import Planetoid
from pathlib import Path
from .registry import register_dataset
from .base import BaseDataset
from .utils import tg_to_polars

@register_dataset("Cora")
class CoraDataset(BaseDataset):
    """
    Cora dataset from the Planetoid dataset collection.
    """
    def __init__(self, root: str | None=None, **kwargs):
        """
        Initialize the Cora dataset.
        """
        logging.basicConfig(level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data" / "cora")
        super().__init__(root, **kwargs)
        kwargs.setdefault("name", "Cora")
        data = Planetoid(root, **kwargs)
        self.data = data[0]
        self.edge_df, self.node_df = tg_to_polars(data)
        self.raphtory_graph = self._to_raphtory()