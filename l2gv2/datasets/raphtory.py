import torch
from torch_geometric.data import Data
from raphtory import Graph
from typing import List, Dict


def tg_to_raphtory(data_objects: List[Data]) -> Graph:
    graph = Graph()
    
    for timestamp, data in enumerate(data_objects):
        # Add nodes
        for node_idx in range(data.num_nodes):
            node_id = str(node_idx)
            properties = {}
            if data.x is not None:
                properties = {f"feature_{i}": float(value) for i, value in enumerate(data.x[node_idx])}
            graph.add_node(timestamp=timestamp, id=node_id, properties=properties)
        
        # Add edges
        edge_index = data.edge_index.t().tolist()
        for idx, (src, dst) in enumerate(edge_index):
            properties = {}
            if data.edge_attr is not None:
                properties = {f"feature_{i}": float(value) for i, value in enumerate(data.edge_attr[idx])}
            graph.add_edge(timestamp=timestamp, src=str(src), dst=str(dst), properties=properties)
    
    return graph

def raphtory_to_tg(graph: Graph) -> List[Data]:
    data_objects = []
    
    for timestamp in range(graph.min_time, graph.max_time + 1):
        view = graph.time_view(timestamp)
        
        # Get nodes and their features
        nodes = view.nodes()
        num_nodes = len(nodes)
        node_id_to_idx = {node.id: idx for idx, node in enumerate(nodes)}
        
        node_features = []
        for node in nodes:
            features = [node.properties.get(f"feature_{i}", 0) for i in range(100)]  # Adjust range as needed
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float) if node_features else None
        
        # Get edges and their features
        edges = view.edges()
        edge_index = []
        edge_attr = []
        
        for edge in edges:
            src_idx = node_id_to_idx[edge.src.id]
            dst_idx = node_id_to_idx[edge.dst.id]
            edge_index.append([src_idx, dst_idx])
            
            features = [edge.properties.get(f"feature_{i}", 0) for i in range(100)]  # Adjust range as needed
            edge_attr.append(features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        data_objects.append(data)
    
    return data_objects
