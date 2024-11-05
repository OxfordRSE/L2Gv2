"""Module with functions for the optimization of the 
embeddings of the patches using the manopt library."""

import random
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import local2global as l2g
import numpy as np


def double_intersections_nodes(patches):
    """Returns a dictionary with the nodes that are shared by two patches.

    Args:
        patches: list of patches.

    Returns:
        double_intersections: dictionary with the nodes that are shared by two patches.
    """

    double_intersections = {}
    for i, patch in enumerate(patches):
        for j in range(i + 1, len(patches)):
            double_intersections[(i, j)] = list(
                set(patch.nodes.tolist()).intersection(
                    set(patches[j].nodes.tolist())
                )
            )
    return double_intersections


def anp_loss_nodes_consecutive_patches(
    rotations, scales, translations, patches, nodes, dim, random_choice=True
):
    """Returns the loss function for the optimization of the embeddings of the patches.

    Args:
        rotations: list of orthogonal matrices for embeddings of patches.

        scales: list of scales for embeddings of patches.
        translations: list of translations for embeddings of patches.

        patches: list of patches.

        nodes: dictionary with the nodes that are shared by two patches.

        dim: dimension of the embeddings.

        random_choice: if True, the loss function is computed only for 
        a random subset of nodes shared by two patches.

    Returns:
        loss_function: loss function.
    """

    loss_function = 0
    for i in range(len(patches) - 1):
        if random_choice:
            for n in random.sample(nodes[i, i + 1], dim + 1):
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                loss_function += anp.linalg.norm(theta1 - theta2) ** 2
        else:
            for n in nodes[i, i + 1]:
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                loss_function += anp.linalg.norm(theta1 - theta2) ** 2

    return loss_function


def optimization(patches, nodes, dim):
    """ Returns the embeddings of the patches using the manopt library.

    Args:
        patches: list of patches.

        nodes: dictionary with the nodes that are shared by two patches.

        dim: dimension of the embeddings.

    Returns:
        result: result of the optimization.

        embedding: embeddings of the patches.
    """
    n_patches = len(patches)

    anp.random.seed(42)

    od = [pymanopt.manifolds.SpecialOrthogonalGroup(dim) for i in range(n_patches)]
    rd = [pymanopt.manifolds.Euclidean(dim) for i in range(n_patches)]
    r1 = [pymanopt.manifolds.Euclidean(1) for i in range(n_patches)]
    prod = od + rd + r1

    manifold = pymanopt.manifolds.product.Product(prod)

    @pymanopt.function.autograd(manifold)
    def cost(*R):
        rs = list(R[:n_patches])
        ts = list(R[n_patches:2 * n_patches])
        ss = list(R[2 * n_patches:])
        return anp_loss_nodes_consecutive_patches(rs, ss, ts, patches, nodes, dim)

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem, reuse_line_searcher=True)

    rotations = result.point[:n_patches]

    translations = result.point[n_patches : 2 * n_patches]

    scales = result.point[2 * n_patches :]
    emb_problem = l2g.AlignmentProblem(patches)

    embedding = np.empty((emb_problem.n_nodes, emb_problem.dim))
    for node, patch_list in enumerate(emb_problem.patch_index):
        embedding[node] = np.mean(
            [
                scales[i] * emb_problem.patches[p].get_coordinate(node) @ rotations[i]
                + translations[i]
                for i, p in enumerate(patch_list)
            ],
            axis=0,
        )

    return result, embedding
