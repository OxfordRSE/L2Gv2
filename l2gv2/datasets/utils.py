"""
Utilities for converting between different graph formats.
"""
import numpy as np
import polars as pl
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, Callable, List
import torch
from torch import Tensor
from torch_geometric.utils import coalesce
from raphtory import Graph

def polars_to_tg(edge_df: pl.DataFrame, node_df: pl.DataFrame=None, pre_transform: Optional[Callable] = None) ->  Tuple[Data, Optional[Dict[str, Tensor]]]:
    """
    Convert a pair of Polars DataFrames (edge and node) to a list of PyTorch Geometric Data objects.
    """
    # Process nodes first to get the number of nodes and node features
    num_nodes = max(int(edge_df["src"].max() + 1),
                    int(edge_df["dst"].max() + 1))
    if node_df is not None:
        feature_cols = [col for col in node_df.columns if col not in ["timestamp", "id"]]
        if feature_cols:
            # Group by node id, take first occurrence of features, and sort by id
            node_features = node_df.group_by("id").agg([
                pl.col(col).first() for col in feature_cols
            ]).sort("id").select(feature_cols).to_numpy
            x = torch.from_numpy(node_features)
        else:
            x = None
    else:
        x = None
    # Process edges for each timestamp
    data_list = []
    unique_timestamps = sorted(edge_df["timestamp"].unique().to_list())
    for ts in unique_timestamps:
        # Process edges
        edges_filtered  = edge_df.filter(pl.col("timestamp") == ts).select(["src", "dst"])
        edge_array = edges_filtered.to_numpy()
        if edge_array.size == 0:
            continue
        edge_index = torch.from_numpy(edge_array).t()
        # Process edge features
        edge_feature_cols = [col for col in edges_filtered.columns if col not in ["timestamp", "src", "dst"]]
        if edge_feature_cols:
            edge_features = edges_filtered.select(edge_feature_cols).to_numpy()
            edge_attr = torch.from_numpy(edge_features)
        else:
            edge_attr = None
        if edge_attr is not None:
            edge_index = coalesce(edge_index, edge_attr, num_nodes=num_nodes)
        else:
            edge_index = coalesce(edge_index, num_nodes=num_nodes)
        data = Data(edge_index=edge_index, 
                    num_nodes=num_nodes, 
                    timestamp=ts)
        if x is not None:
            data.x = x
        if edge_attr is not None:
            data.edge_attr = edge_attr
        data_list.append(data)
    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    return data_list

def polars_to_raphtory(edge_df: pl.DataFrame, node_df: pl.DataFrame=None) -> Graph:
    """
    Convert a Polars DataFrame to a Raphtory graph.
    """
    graph = Graph()
    # Convert Polars DataFrames to Pandas DataFrames
    edge_df_pd = edge_df.to_pandas()
    if node_df is not None:
        node_df_pd = node_df.to_pandas()
    else:
        node_df_pd = None

    graph.load_edges_from_pandas(
        edge_df_pd,
        time="timestamp",
        src="src", 
        dst="dst",
        properties=[col for col in edge_df.columns if col not in ["timestamp", "src", "dst"]]
    )
    if node_df is not None:
        graph.load_nodes_from_pandas(
            df=node_df_pd,
            time="timestamp",
            id="id",
            constant_properties=[c for c in node_df.columns if c not in ["timestamp", "id"]]
        )
    return graph

def tg_to_polars(data_list: List[Data]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convert a list of PyTorch Geometric Data objects to a Polars DataFrame.
    """
    edge_dfs = []
    node_dfs = []
    for data in data_list:
        # Handle edges
        edge_index = data.edge_index.t().numpy()
        edge_dict = {
            "src": edge_index[:, 0],
            "dst": edge_index[:, 1]
        }
        if hasattr(data, "timestamp"):
            edge_dict["timestamp"] = data.timestamp
        else:
            edge_dict["timestamp"] = np.int64(0)
        if hasattr(data, "edge_attr"):
            if data.edge_attr is not None:
                edge_attr = data.edge_attr.numpy()
                for i in range(edge_attr.shape[1]):
                    edge_dict[f"edge_feature_{i}"] = edge_attr[:, i]
        edge_dfs.append(pl.DataFrame(edge_dict))

        # Handle nodes
        if hasattr(data, "x"):
            if data.x is not None:
                node_features = data.x.numpy()
                node_dict = {
                    "id": range(data.num_nodes)
                }
        if hasattr(data, "timestamp"):
            node_dict["timestamp"] = data.timestamp
        else:
            node_dict["timestamp"] = np.int64(0)
        for i in range(node_features.shape[1]):
            node_dict[f"node_feature_{i}"] = node_features[:, i]

        # Handle labels
        if hasattr(data, "y"):
            if data.y is not None:
                node_dict["y"] = data.y.numpy()
        
        node_dfs.append(pl.DataFrame(node_dict))

    edge_df = pl.concat(edge_dfs)
    node_df = pl.concat(node_dfs)

    return edge_df, node_df

def tg_to_raphtory(data_list: List[Data]) -> Graph:
    """
    Convert a list of PyTorch Geometric Data objects to a Raphtory graph.
    """
    graph = Graph()
    for idx, data in enumerate(dataset):
        if hasattr(data, "timestamp"):
            timestamp = data.timestamp
        else:
            timestamp = idx
        for node_idx in range(data.num_nodes):
            node_id = str(node_idx)
            properties = {}
            if data.x is not None:
                properties = {f"feature_{i}": float(value) for i, value in enumerate(data.x[node_idx])}
            graph.add_node(timestamp=timestamp, id=node_id, properties=properties)
        
        edge_index = data.edge_index.t().tolist()
        for idx, (src, dst) in enumerate(edge_index):
            properties = {}
            if data.edge_attr is not None:
                properties = {f"feature_{i}": float(value) for i, value in enumerate(data.edge_attr[idx])}
            graph.add_edge(timestamp=timestamp, src=str(src), dst=str(dst), properties=properties)
    
    return graph
