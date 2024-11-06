"""Anomaly detection module."""

import numpy as np


def raw_anomaly_score_node_patch(aligned_patch_emb, emb, node):
    """ TODO: docstring for `raw_anomaly_score_node_patch`

    Args:
        aligned_patch_emb (): 
        
        emb (): 
        
        node (): 

    Returns:
        float: Raw anomaly score of the node in the patch.
    """

    return np.linalg.norm(aligned_patch_emb.get_coordinate(node) - emb[node])


def nodes_in_patches(patch_data: list):
    """ TODO: docstring for `nodes_in_patches`

    Args:
        patch_data (list): 

    Returns:
        list: 
    """

    return [set(p.nodes.numpy()) for p in patch_data]


def normalized_anomaly(patch_emb: list, patch_data: list, emb: np.array):
    """ TODO: docstring for `normalized_anomaly`

    Args:
        patch_emb (list): 
        
        patch_data (list): 
        
        emb (np.array): 

    Returns:
        np.array: 
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


def get_outliers(patch_emb: list, patch_data: list, emb, k: float):
    """ TODO: docstring for `get_outliers`

    Args:
        patch_emb (list): 

        patch_data (list): 
        
        emb (np.array): 
        
        k (float): Threshold for outliers as multiplier of the standard deviation.

    Returns:
        list: 
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
