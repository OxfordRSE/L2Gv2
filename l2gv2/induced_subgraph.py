""" Induced subgraph utility functions. """
import torch
import torch_geometric as tg


def induced_subgraph(data: tg.data.Data, nodes, extend_hops=0):
    """ Induce a subgraph from the given nodes. 
    
    Args:
        data: The original graph data.

        nodes: The nodes to induce the subgraph from.
        
        extend_hops: The number of hops to extend the subgraph.

    Returns:
        The subgraph data object. 
    """

    nodes = torch.as_tensor(nodes, dtype=torch.long)
    if extend_hops > 0:
        nodes, edge_index, _, edge_mask = tg.utils.k_hop_subgraph(
            nodes, num_hops=extend_hops, edge_index=data.edge_index, relabel_nodes=True
        )
        edge_attr = data.edge_attr[edge_mask, :] if data.edge_attr is not None else None
    else:
        edge_index, edge_attr = tg.utils.subgraph(
            nodes, data.edge_index, data.edge_attr, relabel_nodes=True
        )

    subgraph = tg.data.Data(edge_index=edge_index, edge_attr=edge_attr)
    for key, value in data.__dict__.items():
        if not key.startswith("edge"):
            if hasattr(value, "shape") and value.shape[0] == data.num_nodes:
                setattr(subgraph, key, value[nodes])
            else:
                setattr(subgraph, key, value)
    subgraph.nodes = nodes
    subgraph.num_nodes = len(nodes)
    return subgraph
