"""Anomaly detection module."""

import numpy as np


def raw_anomaly_score_node_patch(aligned_patch_emb, emb, node):
    """Compute the raw anomaly score of a node in a patch.

    Args:
        aligned_patch_emb (AlignedEmb): Aligned embedding of the patch.
        
        emb (np.array): Embedding of the graph.
        
        node (int): Node id.

    Returns:
        float: Raw anomaly score of the node in the patch.
    """

    return np.linalg.norm(aligned_patch_emb.get_coordinate(node) - emb[node])


def nodes_in_patches(patch_data):
    """Get the nodes in each patch.

    Args:
        patch_data (list): List of PatchData.

    Returns:
        list: List of sets of nodes in each patch.
    """

    return [set(p.nodes.numpy()) for p in patch_data]


def normalized_anomaly(patch_emb, patch_data, emb):
    """Compute the normalized anomaly score of each node in each patch.

    Args:
        patch_emb (list): List of AlignedEmb.
        
        patch_data (list): List of PatchData.
        
        emb (np.array): Embedding of the graph.

    Returns:
        np.array: Normalized anomaly score of each node in each patch.
    """

    nodes = nodes_in_patches(patch_data)
    numb_nodes = np.shape(emb)[0]
    numb_patches = len(patch_emb)
    st = np.zeros((numb_nodes, numb_patches))
    mu = np.zeros((numb_nodes, numb_patches))
    raw_anomaly = np.zeros((numb_nodes, numb_patches))

    for n in range(numb_nodes):
        for j in range(numb_patches):
            st[n, j] = np.std(
                [
                    raw_anomaly_score_node_patch(patch_emb[i], emb, n)
                    for i in range(numb_patches)
                    if (n in nodes[i]) & (i != j)
                ]
            )
            mu[n, j] = np.mean(
                [
                    raw_anomaly_score_node_patch(patch_emb[i], emb, n)
                    for i in range(numb_patches)
                    if (n in nodes[i]) & (i != j)
                ]
            )
            if n in nodes[j]:
                raw_anomaly[n, j] = raw_anomaly_score_node_patch(patch_emb[j], emb, n)
    final_score = np.zeros((numb_nodes, numb_patches))
    for n in range(numb_nodes):
        for j in range(numb_patches):
            if n in nodes[j]:
                if (
                    (st[n, j] != 0)
                    & (str(mu[n, j]) != "nan")
                    & (str(st[n, j]) != "nan")
                ):
                    final_score[n, j] = (raw_anomaly[n, j] - mu[n, j]) / st[n, j]

    return final_score


def get_outliers(patch_emb, patch_data, emb, k):
    """Get the outliers of the graph.

    Args:
        patch_emb (list): List of AlignedEmb.

        patch_data (list): List of PatchData.
        
        emb (np.array): Embedding of the graph.
        
        k (float): Threshold.

    Returns:
        list: List of outliers.
    """

    out = []
    numb_nodes = np.shape(emb)[0]
    numb_patches = len(patch_emb)
    final_score = normalized_anomaly(patch_emb, patch_data, emb)
    mean = np.mean(final_score)
    std = np.std(final_score)
    for n in range(numb_nodes):
        for j in range(numb_patches):
            if np.abs(final_score[n, j] - mean) >= k * std:
                out.append(n)
    return out
