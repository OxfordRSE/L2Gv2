# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""Module with functions for the optimization of the
embeddings of the patches using the manopt library."""

import random
from typing import Any, Literal
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import numpy as np
from .patch import utils as ut
from .patch.patch import Patch


def double_intersections_nodes(
    patches: list[Patch],
) -> dict[tuple[int, int], list[int]]:
    """TODO: docstring for `double_intersections_nodes`.

    Args:
        patches (list[Patch]): [description]

    Returns:
        dict[tuple[int, int], list[int]]: [description]
    """

    double_intersections = {}
    for i, patch in enumerate(patches):
        for j in range(i + 1, len(patches)):
            double_intersections[(i, j)] = list(
                set(patch.nodes.tolist()).intersection(set(patches[j].nodes.tolist()))
            )
    return double_intersections


def total_loss(
    rotations,
    scales,
    translations,
    nodes,
    patches,
    dim,
    k,
    rand: bool = False,
) -> tuple[np.floating[Any] | float, dict]:
    """TODO: docstring for `total_loss`.

    R: list of orthogonal matrices for embeddings of patches.

    Args:
        rotations: [description]

        scales: [description]

        translations: [description]

        nodes: [description]

        patches: [description]

        dim (int): [description]

        k (int): [description]

        rand: [description], default is False.

    Returns:
        float: [description]
    """

    loss_value = 0
    fij = {}

    for i, p in enumerate(patches):
        for j, q in enumerate(patches[i + 1 :]):
            if rand:
                for n in random.sample(
                    nodes[i, j + i + 1], min(k * dim + 1, len(nodes[i, i + j + 1]))
                ):
                    theta1 = (
                        scales[i] * p.get_coordinate(n) @ rotations[i] + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * q.get_coordinate(n) @ rotations[j + i + 1]
                        + translations[j + i + 1]
                    )
                    loss_value += np.linalg.norm(theta1 - theta2) ** 2

                    fij[(i, j + 1 + i, n)] = [theta1, theta2]

            else:
                for n in nodes[i, j + i + 1]:
                    theta1 = (
                        scales[i] * rotations[i] @ p.get_coordinate(n) + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * rotations[j + i + 1] @ q.get_coordinate(n)
                        + translations[j + i + 1]
                    )
                    loss_value += np.linalg.norm(theta1 - theta2) ** 2

                    fij[(i, j + 1 + i, n)] = [theta1, theta2]

    return 1 / len(patches) * loss_value, fij


def loss(
    rotations,
    scales,
    translations,
    nodes,
    patches,
    dim,
    k,
    consecutive: bool = False,
    random_choice_in_intersections: bool = False,
    fij: bool = False,
) -> (
    tuple[np.floating[Any] | float | Literal[0], dict | None]
    | np.floating[Any]
    | Literal[0]
):
    """TODO: docstring for `loss`.

    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations: [description]

        scales: [description]

        translations: [description]

        nodes: [description]

        patches: [description]

        dim (int): [description]

        k (int): [description]

        consecutive: [description], default is False.

        random_choice_in_intersections: [description], default is False.

        fij: [description], default is False.

    Returns:

        float: [description]

        list[float]: [description]

    """

    if consecutive:
        loss_value, f = consecutive_loss(
            rotations,
            scales,
            translations,
            nodes,
            patches,
            dim,
            k,
            rand=random_choice_in_intersections,
        )
        if fij:
            return loss_value, f

        return loss_value

    loss_value, f = total_loss(
        rotations,
        scales,
        translations,
        nodes,
        patches,
        dim,
        k,
        rand=random_choice_in_intersections,
    )
    if fij:
        return loss_value, f

    return loss_value, None


def consecutive_loss(
    rotations, scales, translations, nodes, patches, dim, k, rand: bool = True
):
    """TODO: docstring for `consecutive_loss`.
    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations: [description]

        scales: [description]

        translations: [description]

        nodes: [description]

        patches: [description]

        dim (int): [description]

        k (int): [description]

        rand: [description], default is True.

    Returns:

            float: [description]

            list[float]: [description]

    """

    loss_value = 0
    fij = {}

    for i in range(len(patches) - 1):
        if rand:
            for n in random.sample(
                nodes[i, i + 1], min(k * dim + 1, len(nodes[i, i + 1]))
            ):
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                loss_value += np.linalg.norm(theta1 - theta2) ** 2

                fij[(i, 1 + i, n)] = [theta1, theta2]
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
                loss_value += np.linalg.norm(theta1 - theta2) ** 2

                fij[(i, 1 + i, n)] = [theta1, theta2]

    return loss_value, fij


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint does not infer autograd.numpy.linalg.norm so disable no-member
def ANPloss_nodes_consecutive_patches(
    rotations, scales, translations, patches, nodes, dim, k, rand: bool = True
):
    """TODO: docstring for `ANPloss_nodes_consecutive_patches`.

    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations: [description]

        scales: [description]

        translations: [description]

        patches: [description]

        nodes: [description]

        dim (int): [description]

        k (int): [description]

        rand: [description], default is True.

    Returns:

        float: [description]

    """
    loss_value = 0
    # fij=dict()
    for i in range(len(patches) - 1):
        if rand:
            for n in random.sample(
                nodes[i, i + 1], min(k * dim + 1, len(nodes[i, i + 1]))
            ):
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                loss_value += anp.linalg.norm(theta1 - theta2) ** 2
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
                loss_value += anp.linalg.norm(theta1 - theta2) ** 2

    return loss_value  # , fij


# pylint: enable=invalid-name
# pylint: enable=no-member


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint does not infer autograd.numpy.linalg.norm so disable no-member
def ANPloss_nodes(
    rotations, scales, translations, patches, nodes, dim, k, rand: bool = True
):
    """TODO: docstring for `ANPloss_nodes`.
    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations: [description]

        scales: [description]

        translations: [description]

        patches: [description]

        nodes: [description]

        dim (int): [description]

        k (int): [description]

        rand: [description], default is True.

    Returns:

        float: [description]

    """
    loss_value = 0
    # fij=dict()

    for i, p in enumerate(patches):
        for j, q in enumerate(patches[i + 1 :]):
            if rand:
                for n in random.sample(
                    nodes[i, j + i + 1], min(k * dim + 1, len(nodes[i, j + 1 + i]))
                ):
                    theta1 = (
                        scales[i] * p.get_coordinate(n) @ rotations[i] + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * q.get_coordinate(n) @ rotations[j + i + 1]
                        + translations[j + i + 1]
                    )
                    loss_value += anp.linalg.norm(theta1 - theta2) ** 2

                    # fij[(i, j+1+i, n)]=[theta1, theta2]

            else:
                for n in nodes[i, j + i + 1]:
                    theta1 = (
                        scales[i] * rotations[i] @ p.get_coordinate(n) + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * rotations[j + i + 1] @ q.get_coordinate(n)
                        + translations[j + i + 1]
                    )
                    loss_value += anp.linalg.norm(theta1 - theta2) ** 2

                    # fij[(i, j+1+i, n)]=[theta1, theta2]

    return 1 / len(patches) * loss_value  # fij


# pylint enable=invalid-name
# pylint enable=no-member


# pylint: disable=no-member
# pylint does not infer autograd.numpy.random.seed so disable no-member
def optimization(
    patches,
    nodes,
    k,
    consecutive: bool = True,
    random_choice: bool = True,
):
    """TODO: docstring for `optimization`.

    Args:

        patches: [description]

        nodes: [description]

        k: [description]

        consecutive: [description], default is True.

        random_choice: [description], default is True.

    Returns:

        [type]: [description]

        [type]: [description]

    """
    n_patches = len(patches)
    dim = np.shape(patches[0].coordinates)[1]

    anp.random.seed(42)

    od = [pymanopt.manifolds.SpecialOrthogonalGroup(dim) for i in range(n_patches)]
    rd = [pymanopt.manifolds.Euclidean(dim) for i in range(n_patches)]
    r1 = [pymanopt.manifolds.Euclidean(1) for i in range(n_patches)]
    prod = od + rd + r1

    manifold = pymanopt.manifolds.product.Product(prod)

    if consecutive:

        @pymanopt.function.autograd(manifold)
        def cost(*R):
            rs = list(R[:n_patches])
            ts = list(R[n_patches : 2 * n_patches])
            ss = list(R[2 * n_patches :])
            return ANPloss_nodes_consecutive_patches(
                rs, ss, ts, patches, nodes, dim, k, rand=random_choice
            )
    else:

        @pymanopt.function.autograd(manifold)
        def cost(*R):
            rs = list(R[:n_patches])
            ts = list(R[n_patches : 2 * n_patches])
            ss = list(R[2 * n_patches :])
            return ANPloss_nodes(rs, ss, ts, patches, nodes, dim, k, rand=random_choice)

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem, reuse_line_searcher=True)

    rotations = result.point[:n_patches]

    translations = result.point[n_patches : 2 * n_patches]

    scales = result.point[2 * n_patches :]
    emb_problem = ut.AlignmentProblem(patches)

    if emb_problem.n_nodes is None or emb_problem.dim is None:
        raise ValueError("Both n_nodes and dim must be set to integer values.")

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


# pylint enable=no-member


def loss_dictionary(rs, ss, ts, nodes, patches, dim, k):
    """TODO: docstring for `loss_dictionary`.

    Args:

        Rs: [description]

        ss: [description]

        ts: [description]

        nodes: [description]

        patches: [description]

        dim (int): [description]

        k (int): [description]

    Returns:

            [type]: [description]

    """
    loss_dict = {}
    for i in range(2):
        for j in range(2):
            loss_dict[i, j] = loss(
                rs,
                ss,
                ts,
                nodes,
                patches,
                dim,
                k,
                consecutive=(i > 0),
                random_choice_in_intersections=(j > 0),
                fij=False,
            )
    return loss_dict
