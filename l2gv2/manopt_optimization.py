"""Module with functions for the optimization of the
embeddings of the patches using the manopt library."""

import random
from typing import Tuple
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import numpy as np
import local2global as l2g

from l2gv2.patch.patch import Patch


def double_intersections_nodes(
    patches: list[Patch],
) -> dict[tuple[int, int], list[int]]:
    """TODO: docstring for `double_intersections_nodes`.

    Args:
        patches (list[Patch]):

    Returns:
        dict[tuple[int, int], list[int]]:
    """

    double_intersections = {}
    for i, patch in enumerate(patches):
        for j in range(i + 1, len(patches)):
            double_intersections[(i, j)] = list(
                set(patch.nodes.tolist()).intersection(set(patches[j].nodes.tolist()))
            )
    return double_intersections


def anp_loss_nodes_consecutive_patches(
    rotations,
    scales,
    translations,
    patches,
    nodes,
    dim: int,
    random_choice: bool = True,
) -> float:
    """TODO: docstring for `anp_loss_nodes_consecutive_patches`.

    Args:
        rotations ():

        scales ():

        translations (int):

        patches ():

        nodes ():

        dim (int):

        random_choice (bool, optional): default is True.

    Returns:
        float: loss function.
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


def optimization(
    patches: list[Patch], nodes, dim: int
) -> Tuple[pymanopt.OptimizationResult, np.ndarray]:
    """TODO: docstring for `optimization`.

    Args:
        patches ():

        nodes ():

        dim (int):

    Returns:
        result:

        embedding:
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
        ts = list(R[n_patches : 2 * n_patches])
        ss = list(R[2 * n_patches :])
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
