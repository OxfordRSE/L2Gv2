"""Tests for l2gv2.graphs.graph"""

import pytest
import numpy as np
import networkx as nx

from l2gv2.graphs.graph import Graph


def test_from_networkx_directed_with_weights():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=2.0)

    graph = Graph.from_networkx(G, weight="weight")

    expected_edges = np.array([[0, 1], [1, 2]])

    np.testing.assert_array_equal(graph.edge_index, expected_edges.T)
    assert graph.weighted
    assert graph.num_nodes == 3
    assert graph.num_edges == 2
    assert graph.undir is False


def test_from_networkx_undirected_no_weights():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    graph = Graph.from_networkx(G)

    # Edges appear in both directions in directed view
    expected_edges = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])

    np.testing.assert_array_equal(graph.edge_index, expected_edges)
    assert not graph.weighted
    assert graph.num_nodes == 3
    assert graph.num_edges == 4
    assert graph.undir is True


def test_missing_weights_raises():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2)  # Missing weight

    with pytest.raises(RuntimeError, match="some edges have missing weight"):
        Graph.from_networkx(G, weight="weight")
