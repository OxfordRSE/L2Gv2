"""
Graph clustering algorithms

This module contains functions for clustering graphs.

TODO:
- Make the algorithms work with Graph objects
"""

from raphtory import Graph  # pylint: disable=no-name-in-module
from collections.abc import Iterable
from typing import Sequence, Callable


import community
import torch
import pymetis
import numpy as np
import numba
from tqdm import tqdm
from torch_geometric.data import Data

from l2gv2.graphs import TGraph
from l2gv2.utils import tqdm_close


def hierarchical_clustering(
    data: Data, m: int, k: int, clustering_function: Callable[[Data, int], torch.Tensor]
) -> list[torch.Tensor]:
    """
    Perform hierarchical clustering on a PyTorch Geometric graph.

    Args:
        data (Data): The input PyTorch Geometric graph.
        m (int): Target number of clusters.
        k (int): Target maximum cluster size.
        clustering_function (Callable): A function that takes a Data object and the number of clusters `m`
                                         and returns a cluster assignment tensor.

    Returns:
        List[torch.Tensor]: A list of cluster assignment tensors for all levels of the hierarchy.
    """

    def recursive_clustering(data, m, k):
        # Apply the clustering function to get initial clusters
        cluster_tensor = clustering_function(data, m)

        # Check the size of each cluster
        unique_clusters, _ = torch.unique(cluster_tensor, return_counts=True)

        # Store the final cluster assignments
        final_clusters = []

        for cluster_id in unique_clusters:
            # Get nodes in this cluster
            cluster_nodes = (cluster_tensor == cluster_id).nonzero(as_tuple=True)[0]

            if len(cluster_nodes) <= k:
                # If the cluster size is within the limit, keep it as-is
                final_clusters.append((cluster_id, cluster_nodes))
            else:
                # If the cluster size exceeds k, extract the subgraph and apply recursion
                subgraph = extract_subgraph(data, cluster_nodes)
                sub_clusters = recursive_clustering(subgraph, m, k)

                # Adjust sub-cluster IDs to avoid clashes with existing IDs
                max_cluster_id = max(unique_clusters).item()
                sub_clusters_adjusted = [
                    (max_cluster_id + 1 + i, nodes)
                    for i, (_, nodes) in enumerate(sub_clusters)
                ]
                final_clusters.extend(sub_clusters_adjusted)

        return final_clusters

    def extract_subgraph(data, node_indices):
        """
        Extract a subgraph containing only the specified nodes.

        Args:
            data (Data): The original graph.
            node_indices (Tensor): Indices of nodes to include in the subgraph.

        Returns:
            Data: The extracted subgraph.
        """
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[node_indices] = True

        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        edge_index = data.edge_index[:, edge_mask]

        # Reindex nodes in the subgraph
        node_mapping = {
            old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)
        }
        edge_index = torch.tensor(
            [
                [node_mapping[src.item()], node_mapping[dst.item()]]
                for src, dst in edge_index.t()
            ],
            dtype=torch.long,
        ).t()

        return Data(edge_index=edge_index, num_nodes=len(node_indices))

    # Start recursive clustering from the root level
    return recursive_clustering(data, m, k)


def fennel_clustering(
    graph: Graph,
    num_clusters,
    load_limit=1.1,
    alpha=None,
    gamma=1.5,
    num_iters=1,
    clusters=None,
):
    if clusters is None:
        clusters = _fennel_clustering(
            graph.edge_index,
            graph.adj_index,
            graph.num_nodes,
            num_clusters,
            load_limit,
            alpha,
            gamma,
            num_iters,
        )
    else:
        clusters = _fennel_clustering(
            graph.edge_index,
            graph.adj_index,
            graph.num_nodes,
            num_clusters,
            load_limit,
            alpha,
            gamma,
            num_iters,
            clusters,
        )
    return torch.as_tensor(clusters)


@numba.njit
def _fennel_clustering(
    edge_index: np.ndarray,
    adj_index: np.ndarray,
    num_nodes: int,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float | None = None,
    gamma: float = 1.5,
    num_iters: int = 1,
    clusters=np.empty(0, dtype=np.int64),
):
    r"""
    FENNEL single-pass graph clustering algorithm

    Implements the graph clustering algorithm of [#fennel]_.

    Args:
        graph: input graph
        num_clusters: target number of clusters
        load_limit: maximum cluster size is ``load_limit * graph.num_nodes / num_clusters`` (default: ``1.1``)
        alpha: :math:`\alpha` value for the algorithm (default as suggested in [#fennel]_)
        gamma: :math:`\gamma` value for the algorithm (default: 1.5)
        randomise_order: if ``True``, randomise order, else use breadth-first-search order.
        clusters: input clustering to refine (optional)
        num_iters: number of cluster assignment iterations (default: ``1``)

    Returns:
        cluster index tensor

    References:
        .. [#fennel] C. Tsourakakis et al. “FENNEL: Streaming Graph Partitioning for Massive Scale Graphs”.
                     In: Proceedings of the 7th ACM international conference on Web search and data mining.
                     WSDM'14 (2014) doi: `10.1145/2556195.2556213 <https://doi.org/10.1145/2556195.2556213>`_.

    """
    num_edges = edge_index.shape[1]

    if alpha is None:
        alpha = num_edges * (num_clusters ** (gamma - 1)) / (num_nodes**gamma)

    partition_sizes = np.zeros(num_clusters, dtype=np.int64)
    if clusters.size == 0:
        clusters = np.full((num_nodes,), -1, dtype=np.int64)
    else:
        # There is already a clustering, so we need to copy it and update the partition sizes
        clusters = np.copy(clusters)
        for index in clusters:
            partition_sizes[index] += 1

    # Maximum number of nodes per cluster
    load_limit *= num_nodes / num_clusters

    assert alpha
    deltas = -alpha * gamma * (partition_sizes ** (gamma - 1))

    with numba.objmode:
        progress = tqdm(total=num_nodes)

    for it in range(num_iters):
        not_converged = 0

        progress_it = 0
        for i in range(num_nodes):
            cluster_indices = np.empty(
                (adj_index[i + 1] - adj_index[i],), dtype=np.int64
            )
            for ni, index in enumerate(range(adj_index[i], adj_index[i + 1])):
                cluster_indices[ni] = clusters[edge_index[1, index]]
            old_cluster = clusters[i]
            if old_cluster >= 0:
                partition_sizes[old_cluster] -= 1
            cluster_indices = cluster_indices[cluster_indices >= 0]

            if cluster_indices.size > 0:
                c_size = np.zeros(num_clusters, dtype=np.int64)
                for index in cluster_indices:
                    c_size[index] += 1
                ind = np.argmax(deltas + c_size)
            else:
                ind = np.argmax(deltas)
            clusters[i] = ind
            partition_sizes[ind] += 1
            if partition_sizes[ind] == load_limit:
                deltas[ind] = -np.inf
            else:
                deltas[ind] = -alpha * gamma * (partition_sizes[ind] ** (gamma - 1))
            not_converged += ind != old_cluster

            if i % 10000 == 0 and i > 0:
                progress_it = i
                with numba.objmode:
                    progress.update(10000)
        with numba.objmode:
            progress.update(num_nodes - progress_it)

        print(f"iteration: {str(it)}, not converged: {str(not_converged)}")

        if not_converged == 0:
            print(f"converged after {str(it)} iterations.")
            break
    with numba.objmode:
        tqdm_close(progress)

    return clusters


def louvain_clustering(graph: Graph, *args, **kwargs):
    r"""
    Implements clustering using the Louvain [#l]_ algorithm for modularity optimisation

    Args:
        graph: input graph

    Returns:
        partition tensor

    This is a minimal wrapper around :py:func:`community.best_partition` from the
    `python-louvain <https://github.com/taynaud/python-louvain>`_ package. Any other
    arguments provided are passed through.

    References:
        .. [#l] V. D. Blondel et al.
                “Fast unfolding of communities in large networks”.
                Journal of Statistical Mechanics: Theory and Experiment 2008.10 (2008), P10008.
                DOI: `10.1088/1742-5468/2008/10/P10008 <https://doi.org/10.1088/1742-5468/2008/10/P10008>`_

    """
    clusters = community.best_partition(
        graph.to_networkx().to_undirected(), *args, **kwargs
    )
    return torch.tensor([clusters[i] for i in range(graph.num_nodes)], dtype=torch.long)


def metis_clustering(graph: TGraph, num_clusters):
    """
    Implements clustering using metis

    Args:
        graph: input graph
        num_clusters: number of cluster

    Returns:
        partition tensor

    This uses the `pymetis <https://github.com/inducer/pymetis>`_ package

    References:
        .. [#metis] “A Fast and Highly Quality Multilevel Scheme for Partitioning Irregular Graphs”.
                    George Karypis and Vipin Kumar.
                    SIAM Journal on Scientific Computing, Vol. 20, No. 1, pp. 359—392, 1999.
    """
    _, memberships = pymetis.part_graph(
        num_clusters,
        adjncy=graph.edge_index[1],
        xadj=graph.adj_index,
        eweights=graph.edge_attr,
    )
    return torch.as_tensor(memberships, dtype=torch.long, device=graph.device)


def spread_clustering(graph: Graph, num_clusters, max_degree_init=True):
    clusters = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.device)
    if max_degree_init:
        seeds = torch.topk(torch.as_tensor(graph.degree), k=num_clusters).indices
    else:
        seeds = torch.multinomial(
            torch.as_tensor(graph.degree), num_clusters, replacement=False
        )

    clusters[seeds] = torch.arange(num_clusters)
    spread_weights = torch.zeros(
        (num_clusters, graph.num_nodes), dtype=torch.double, device=graph.device
    )
    spread_weights[:, seeds] = -1
    unassigned = clusters < 0
    for seed in seeds:
        c = clusters[seed]
        inds, weights = graph.adj_weighted(seed)
        keep = unassigned[inds]
        spread_weights[c, inds[keep]] += weights[keep] / graph.strength[inds[keep]]

    num_unassigned = graph.num_nodes - num_clusters

    while num_unassigned > 0:
        progress = False
        for c in range(num_clusters):
            node = torch.argmax(spread_weights[c])
            if spread_weights[c, node] > 0:
                progress = True
                # make sure node is actually connected to cluster
                clusters[node] = c
                spread_weights[:, node] = -1  # should not be chosen again
                unassigned[node] = False
                num_unassigned -= 1
                inds, weights = graph.adj_weighted(node)
                keep = unassigned[inds]
                spread_weights[c, inds[keep]] += (
                    weights[keep] / graph.strength[inds[keep]]
                )
        if not progress:
            print("increasing number of clusters due to disconnected components")
            unassigned_nodes = torch.nonzero(unassigned).ravel()
            if max_degree_init:
                seed = unassigned_nodes[
                    torch.argmax(torch.as_tensor(graph.degree[unassigned_nodes]))
                ]
            else:
                seed = unassigned_nodes[
                    torch.multinomial(
                        torch.as_tensor(graph.degree[unassigned_nodes]), 1
                    )
                ]
            clusters[seed] = num_clusters
            spread_weights = torch.cat(
                (
                    spread_weights,
                    torch.zeros(
                        (1, graph.num_nodes), dtype=torch.double, device=graph.device
                    ),
                )
            )
            unassigned[seed] = False
            spread_weights[:, seed] = -1
            inds, weights = graph.adj_weighted(seed)
            keep = unassigned[inds]
            spread_weights[num_clusters, inds[keep]] += (
                weights[keep] / graph.strength[inds[keep]]
            )
            num_clusters += 1
            num_unassigned -= 1
    return clusters


def hierarchical_aglomerative_clustering(
    graph: Graph, method=spread_clustering, levels=None, branch_factors=None
):
    r"""
    Hierarchical agglomerative clustering

    Implements a hierarchical agglomerative clustering algorithm.

    Args:
        graph: input graph
        method: clustering method
        levels: number of levels
        branch_factors: branch factors

    Returns:
        list of clusters
    """
    if branch_factors is None:
        branch_factors = [graph.num_nodes ** (1 / (levels + 1)) for _ in range(levels)]
    else:
        if not isinstance(branch_factors, Iterable):
            branch_factors = [branch_factors] * (levels)
        else:
            if levels is None:
                levels = len(branch_factors)
            elif len(branch_factors) != levels:
                raise ValueError(f"{levels=} does not match {len(branch_factors)=}")
    num_clusters = np.cumprod(branch_factors)[::-1]
    clusters = []
    rgraph = graph
    for c in num_clusters:
        cluster = method(rgraph, int(c))
        rgraph = rgraph.partition_graph(cluster)
        clusters.append(cluster)
    return clusters


class Partition(Sequence):
    "Defines a partition of a graph"

    def __init__(self, partition_tensor):
        partition_tensor = torch.as_tensor(partition_tensor)
        counts = torch.bincount(partition_tensor)
        self.num_parts = len(counts)
        self.nodes = torch.argsort(partition_tensor)
        self.part_index = torch.zeros(self.num_parts + 1, dtype=torch.long)
        self.part_index[1:] = torch.cumsum(counts, dim=0)

    def __getitem__(self, item):
        return self.nodes[self.part_index[item] : self.part_index[item + 1]]

    def __len__(self):
        return self.num_parts
