"""Tests for l2gv2.graphs.tgraph"""

import torch


def test_edges(toy_tgraph):
    edges = list(toy_tgraph.edges())
    assert (0, 1) in edges
    assert (2, 0) in edges
    assert len(edges) == toy_tgraph.num_edges


def test_edges_weighted(toy_tgraph):
    edges = list(toy_tgraph.edges_weighted())
    assert len(edges) == toy_tgraph.num_edges


def test_is_edge(toy_tgraph):
    assert toy_tgraph.is_edge(0, 1)
    assert toy_tgraph.is_edge(2, 0)
    assert not toy_tgraph.is_edge(0, 2)


def test_neighbourhood(toy_tgraph):
    nodes = torch.tensor([0])
    neigh = toy_tgraph.neighbourhood(nodes, hops=2)
    assert set(neigh.tolist()) == {0, 1, 2}


def test_subgraph(toy_tgraph):
    nodes = torch.tensor([0, 1])
    sub = toy_tgraph.subgraph(nodes, relabel=True)
    assert sub.num_nodes == 2
    assert all(n in [0, 1] for n, _ in sub.edges())


def test_connected_component_ids(toy_tgraph):
    cc_ids = toy_tgraph.connected_component_ids()
    assert cc_ids.shape == (toy_tgraph.num_nodes,)
    assert cc_ids.min().item() == 0


def test_nodes_in_lcc(toy_tgraph):
    lcc_nodes = toy_tgraph.nodes_in_lcc()
    assert set(lcc_nodes.tolist()) == {0, 1, 2}


def test_to_networkx(toy_tgraph):
    nxg = toy_tgraph.to_networkx()
    assert len(nxg.nodes) == 3


def test_bfs_order(toy_tgraph):
    order = toy_tgraph.bfs_order(start=0)
    assert set(order.tolist()) == {0, 1, 2}


def test_partition_graph(toy_tgraph):
    partition = torch.tensor([0, 0, 1])
    part_g = toy_tgraph.partition_graph(partition)
    assert part_g.num_nodes == 2
    assert part_g.edge_index.shape[0] == 2


def test_sample_positive_edges(toy_tgraph):
    samples = toy_tgraph.sample_positive_edges(2)
    assert samples.shape == (2, 2)


def test_sample_negative_edges(toy_tgraph):
    samples = toy_tgraph.sample_negative_edges(2)
    assert samples.shape == (2, 2)
