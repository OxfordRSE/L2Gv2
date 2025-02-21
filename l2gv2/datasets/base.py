"""
Utilities for loading graph datasets.

The module provides a DataLoader class that can load graph datasets torch-geometric.data.Dataset 
and return a polars DataFrame of the edges and nodes. It contains methods to convert the graph
into a raphtory graph and a networkx graph.
"""

import logging
from pathlib import Path
import polars as pl
from raphtory import Graph
from torch_geometric.data import Data, InMemoryDataset
from torch import Tensor
from typing import Optional, Callable, Tuple, Dict

from .utils import polars_to_tg, polars_to_raphtory
from l2gv2.datasets import DATASET_REGISTRY
datasets = list(DATASET_REGISTRY.keys())

class BaseDataset(InMemoryDataset):
    """
    Wrapper for a PyTorch Geometric Dataset.
    """
    
    def __init__(self, root: str | None=None, transform: Optional[Callable]=None, pre_transform: Optional[Callable]=None):
        """
        Initialize a new BaseDataset instance.

        Args:
            root (str or Path): The root directory where the dataset is stored.
            transform (callable, optional): A function to apply transformations to the data.
            pre_transform (callable, optional): A function to apply preprocessing transformations before the main transform.
        """
        logging.basicConfig(level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / 'raw')

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / 'processed')
    
    def _load_polars(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load the processed edge and node Polars DataFrames.
        """
        if hasattr(self, 'edge_df') and hasattr(self, 'node_df'):
            print("Loading edge and node data from memory")
            return self.edge_df, self.node_df
        processed_dir = Path(self.processed_dir)
        if not Path(processed_dir / "edge_data.parquet").exists():
            self.logger.error(f"Parquet file {processed_dir / "edge_data.parquet"} not found.")
            raise FileNotFoundError(f"Parquet file {processed_dir / "edge_data.parquet"} not found.")
        edge_df = pl.read_parquet(processed_dir / "edge_data.parquet")
        if Path(processed_dir / "node_data.parquet").exists():
            node_df = pl.read_parquet(processed_dir / "node_data.parquet")
        else:
            node_df = None
        return edge_df, node_df
    
    def _to_raphtory(self) -> Graph:
        """
        Convert the processed edge and node Polars DataFrames to a Raphtory graph.
        """
        edge_df, node_df = self._load_polars()
        graph = polars_to_raphtory(edge_df, node_df)
        return graph
        
    def _to_torch_geometric(self) ->  Tuple[Data, Optional[Dict[str, Tensor]]]:
        """
        Convert the processed edge and node Polars DataFrames to a PyTorch Geometric dataset.
        """
        edge_df, node_df = self._load_polars()
        data, slices = self.collate(polars_to_tg(edge_df, node_df, self.pre_transform))
        return data, slices
    
    def to(self, format: str):
        """
        Convert the dataset to a different format.
        """
        if format == "raphtory":
            return self.raphtory_graph
        elif format == "polars":
            return self.edge_df, self.node_df
        else:
            return super().to(format)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"