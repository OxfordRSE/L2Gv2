"""
Utilities for loading graph datasets.

The module provides a DataLoader class that can load graph datasets torch-geometric.data.Dataset 
and return a polars DataFrame of the edges and nodes. It contains methods to convert the graph
into a raphtory graph and a networkx graph.
"""

import datetime
from pathlib import Path
import polars as pl
import torch
import logging
import networkx as nx
import raphtory as rp
import torch_geometric.data
from torch_geometric.data import Dataset
from tqdm import tqdm

from l2gv2.datasets import DATASET_REGISTRY

datasets = list(DATASET_REGISTRY.keys())

EDGE_COLUMNS = {"source", "dest"}  # required columns

EdgeList = list[tuple[str, str]]

def is_graph_dataset(p: Path) -> bool:
    "Returns True if dataset represented by `p` is a valid dataset"
    return (p / (p.stem + "_edges.parquet")).exists()

class GraphDataLoader:
    """
    A class for loading graph datasets and returning a polars DataFrame of the edges and nodes.
    """
    def __init__(self, dset: Dataset):
        """
        Initialize the GraphDataLoader.

        Args:
            dset (Dataset): The dataset to load.
            timestamp_fmt (str, optional): The format of the timestamp. Defaults to "%Y-%m-%d".
        """
        logging.basicConfig(level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

        if not isinstance(self.dset, Dataset):
            raise ValueError("dset must be a torch_geometric.data.Dataset")
        
        self.data = dset

        if hasattr(self.data, "timestamp"):
            self.temporal = True
            self.datelist = self.data.timestamp
        else:
            self.temporal = False
            self.datelist = None
        
        self._parse()
    
    def _parse(self):
        """Parse the dataset into a polars DataFrame"""

        dfs = []
        for slice_data in self.data:
            if self.temporal:
                ts = slice_data.timestamp
            else:
                ts = 0

            edge_index = (
                slice_data.edge_index.numpy() 
                if isinstance(slice_data.edge_index, torch.Tensor) 
                else slice_data.edge_index
            )
            num_edges = edge_index.shape[1]
    
            df_dict = {'timestamp': [ts] * num_edges,
                       'source': edge_index[0],
                       'dest': edge_index[1]
                       }

            if hasattr(slice_data, "edge_attr"):
                df_dict["edge_attr"] = slice_data.edge_attr
                edge_attr = (
                    slice_data.edge_attr.numpy() 
                    if isinstance(slice_data.edge_attr, torch.Tensor) 
                    else slice_data.edge_attr
                )
            # Handle multi-dimensional edge features (e.g., multiple features per edge).
            if edge_attr.ndim == 2:
                num_features = edge_attr.shape[1]
                for i in range(num_features):
                    df_dict[f'edge_feature_{i}'] = edge_attr[:, i]
            else:
                # If there's only one feature per edge, edge_attr may be 1D.
                df_dict['edge_feature'] = edge_attr
                dfs.append(pl.DataFrame(df_dict))
        
        if len(dfs) > 1:
            self.edges = pl.concat(dfs)
        else:
            self.edges = dfs[0]        
        
        # Process nodes
        if hasattr(self.data, 'x') and self.data.x is not None:
            # Convert to a NumPy array if it's a torch.Tensor
            x = self.data.x.numpy() if isinstance(self.data.x, torch.Tensor) else self.data.x

            num_nodes = x.shape[0]
            df_dict = {'node_id': list(range(num_nodes))}

            # If the node feature matrix is 2D (i.e. each node has multiple features)
            if x.ndim == 2:
                num_features = x.shape[1]
                for i in range(num_features):
                    # Create a column for each node feature, e.g., feature_0, feature_1, etc.
                    df_dict[f'feature_{i}'] = x[:, i]
            else:
                # If the feature matrix is 1D, treat it as a single feature column.
                df_dict['feature'] = x

            # Create the Polars DataFrame with the node data.
            self.nodes = pl.DataFrame(df_dict)
        else:
            self.nodes = None

    def get_dates(self) -> list[str]:
        "Returns list of dates"
        return self.datelist.to_list()
    
    def get_edges(self) -> pl.DataFrame:
        "Returns edges as a polars DataFrame"
        return self.edges

    def get_nodes(self, ts: str | None = None) -> pl.DataFrame:
        """Returns node data as a polars DataFrame

        Args:
            ts (str, optional): if specified, only return nodes with this timestamp

        Returns:
            polars.DataFrame
        """
        if ts is None:
            return self.nodes
        if isinstance(ts, str):
            ts = self.timestamp_from_string(ts)
        return self.nodes.filter(pl.col("timestamp") == ts)

    def get_node_list(self, ts: str | None = None) -> list[str]:
        """Returns node list

        Args:
            ts (str, optional): if specified, only return nodes with this timestamp

        Returns:
            list of str
        """
        nodes = self.nodes

        if ts is not None:
            if isinstance(ts, str):
                ts = self.timestamp_from_string(ts)
            nodes = nodes.filter(pl.col("timestamp") == ts)
        return nodes.select("nodes").unique(maintain_order=True).to_series().to_list()

    def get_node_features(self) -> list[str]:
        "Returns node features as a list of strings"
        return self.node_features

    def get_edge_features(self) -> list[str]:
        "Returns edge features as a list of strings"
        return self.edge_features

    def get_graph(self) -> rp.Graph:  # pylint: disable=no-member
        "Returns a raphtory.Graph representation"
        g = rp.Graph()  # pylint: disable=no-member

        g.load_edges_from_pandas(
            df=self.edges.to_pandas(),
            time="timestamp",
            src="source",
            dst="dest",
            properties=self.edge_features,
        )
        g.load_nodes_from_pandas(
            df=self.nodes.to_pandas(),
            time="timestamp",
            id="nodes",
            properties=self.node_features,
        )

        return g

    def get_edge_list(
        self, temp: bool = True
    ) -> EdgeList | dict[datetime.datetime, EdgeList]:
        """Returns edge list

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to edge lists (list of string tuples), if false, returns
                edge list for the entire graph
        """
        if self.temporal and temp:
            edge_list = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list[d] = [tuple(x) for x in edges]
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
        return edge_list

    def get_networkx(
        self, temp: bool = True
    ) -> nx.Graph | dict[datetime.datetime, nx.Graph]:
        """Returns networkx.DiGraph representation

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to networkx digraphs, if false, returns a networkx digraph
        """

        if self.temporal and temp:
            nx_graphs: dict[datetime.datetime, nx.Graph] = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                nx_graphs[d] = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
            return nx_graphs
        edges = self.edges.select("source", "dest").unique().to_numpy()
        edge_list = [tuple(x) for x in edges]
        return nx.from_edgelist(edge_list, create_using=nx.DiGraph)

    def get_edge_index(
        self, temp: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Returns edge index as torch tensors

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to torch tensors (list of string tuples), if false, returns
                a torch tensor.
        """
        if self.temporal and temp:
            edge_index = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                edge_index[d] = (
                    torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                )
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def get_tgeometric(
        self, temp: bool = True
    ) -> torch_geometric.data.Data | dict[datetime.datetime, torch_geometric.data.Data]:
        """Returns torch_geometric representation

        Args:
            temp (bool, optional, default=True): If true, then returns a dictionary of
                timestamps to torch_geometric representations, if false, returns
                a torch_geometric representation.
        """
        nodes = self.nodes.select("nodes").unique().to_numpy()
        features = self.nodes.select(self.node_features).to_numpy()
        if self.temporal and temp:
            tg_graphs = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                tg = torch_geometric.data.Data(edge_index=edge_index)
                tg.nodes = torch.from_numpy(nodes).int()
                tg.x = torch.from_numpy(features).float()
                tg_graphs[d] = tg
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            tg_graphs = torch_geometric.data.Data(edge_index=edge_index)
            tg_graphs.nodes = torch.Tensor(nodes).int()
            tg_graphs.x = torch.from_numpy(features).float()
        return tg_graphs

