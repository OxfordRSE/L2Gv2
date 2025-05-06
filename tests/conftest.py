import pytest
import torch
from l2gv2.graphs import TGraph


@pytest.fixture
def toy_tgraph():
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], dtype=torch.long)
    edge_attr = torch.tensor([1.0, 1.0, 1.0, 1.0])
    return TGraph(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=3,
        undir=True,
        ensure_sorted=True,
    )


@pytest.fixture
def toy_tgraph_weighted():
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], dtype=torch.long)
    edge_attr = torch.tensor([1.0, 2.0, 1.0, 0.5])
    return TGraph(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=3,
        undir=True,
        ensure_sorted=True,
    )
