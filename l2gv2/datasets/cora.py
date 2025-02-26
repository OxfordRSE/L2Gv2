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

    def __init__(self, root: str | None = None, **kwargs):
        """
        Initialize the Cora dataset.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data" / "cora")
        super().__init__(root, **kwargs)
        kwargs.setdefault("name", "Cora")
        data = Planetoid(root, **kwargs)
        self.data = data[0]
        self.edge_df, self.node_df = tg_to_polars(data)
        self.raphtory_graph = self._to_raphtory()

    @property
    def processed_file_names(self) -> str | list[str] | tuple[str, ...]:
        """
        The processed file names for the Cora dataset.
        """
        if not Path(self.processed_dir).exists():
            processed_files = []
        else:
            processed_files = self.data.processed_file_names
        return processed_files

    @property
    def raw_file_names(self) -> list[str]:
        """
        The raw file names for the AS-733 dataset.
        """
        if not Path(self.raw_dir).exists():
            raw_files = []
        else:
            raw_files = self.data.raw_file_names
        return raw_files
