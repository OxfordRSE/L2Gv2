"""
Test functions for l2gv2.datasets
"""

from pathlib import Path

import pytest

from l2gv2.datasets import DataLoader

TEST_DATASETS = Path(__file__).parent / "datasets"
SOCIAL_NODES = [
    "amy",
    "anil",
    "charlie",
    "john",
    "maria",
    "peter",
]
SOCIAL_EDGES = [
    ("amy", "charlie"),
    ("amy", "peter"),
    ("anil", "maria"),
    ("john", "amy"),
    ("maria", "peter"),
    ("peter", "anil"),
]

# disable missing-function-docstring, redefined-outer-name (pytest fixtures)
# pylint: disable=C0116,W0621


def test_dataloader_invalid_dataset():
    with pytest.raises(ValueError):
        DataLoader(TEST_DATASETS / "invalid")


@pytest.fixture
def social_dataset():
    return DataLoader(TEST_DATASETS / "social")


def test_is_temporal(social_dataset):
    assert social_dataset.temporal is True


def test_features(social_dataset):
    assert social_dataset.node_features == ["height_cm"]
    assert social_dataset.edge_features == ["distance_house_km"]


def test_get_dates(social_dataset):
    assert sorted(dt.date().isoformat() for dt in social_dataset.get_dates()) == [
        "2024-05-02",
        "2024-05-03",
    ]


def test_get_edges(social_dataset):
    edges = social_dataset.get_edges()
    source_nodes = edges.select("source").to_series().to_list()
    target_nodes = edges.select("dest").to_series().to_list()
    assert sorted(zip(source_nodes, target_nodes)) == SOCIAL_EDGES


def test_get_nodes(social_dataset):
    nodes = social_dataset.get_nodes()
    assert sorted(nodes.select("nodes").to_series().to_list()) == SOCIAL_NODES


def test_get_graph(social_dataset):
    rpgraph = social_dataset.get_graph()  # raphtory graph format
    assert sorted(rpgraph.nodes.to_df().name.to_list()) == SOCIAL_NODES
    edge_df = rpgraph.edges.to_df()[["src", "dst"]]
    assert sorted(edge_df.itertuples(index=False)) == SOCIAL_EDGES


def test_get_edge_list(social_dataset):
    edges = sorted(social_dataset.get_edge_list(temp=False))
    assert edges == SOCIAL_EDGES


def test_get_networkx(social_dataset):
    nxgraph = social_dataset.get_networkx(temp=False)
    assert sorted(nxgraph.nodes) == SOCIAL_NODES
    assert sorted(nxgraph.edges) == SOCIAL_EDGES


# TODO: add node and edge testing for torch_geometric
