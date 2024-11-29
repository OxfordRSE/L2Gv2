"""Generate synthetic test data."""

import argparse
import csv
from typing import Type
from copy import copy
from os import path
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
import networkx as nx

from .patch import utils as ut
from .patch.patch import Patch


def generate_data(
    n_clusters: int,
    scale: float = 1.0,
    std: float = 0.5,
    max_size: int = 200,
    min_size: int = 10,
    dim: int = 2,
) -> np.ndarray:
    """Generate test data with normally-distributed clusters centered on a sphere.

    Args:
        n_clusters (int): Number of clusters.

        scale: Radius of sphere for cluster centers, default is 1.0].

        std: Standard deviation for cluster points, default is 0.5].

        max_size: Maximum cluster size, default is 200.

        min_size: Minimum cluster size, default is 10.

        dim: Data dimension, default is 2.

    Returns:
        np.ndarray: Generated data points.
    """

    # Random parameters of each cluster
    if dim > 2:
        list_shifts = []
        for _ in range(n_clusters):
            x = ut.rg.normal(size=(1, dim))
            x /= np.linalg.norm(x)
            x *= scale
            list_shifts.append(x)
    elif dim == 2:
        list_shifts = [
            np.array(
                [np.cos(t / n_clusters * 2 * np.pi), np.sin(t / n_clusters * 2 * np.pi)]
            )
            * scale
            for t in range(n_clusters)
        ]
    else:
        raise ValueError("Dimension needs to be >= 2")

    list_var = [std] * n_clusters
    list_sizes = [ut.rg.integers(min_size, max_size) for _ in range(n_clusters)]

    # Make union cluster
    list_of_clusters = [
        ut.rg.normal(scale=1, size=(s, dim)) * v + shift
        for shift, v, s in zip(list_shifts, list_var, list_sizes)
    ]
    return np.vstack(list_of_clusters)


# TODO: fix too-many-branches, too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def voronoi_patches(
    points: np.ndarray,
    sample_size: int = 100,
    min_degree: int | None = None,
    min_overlap: int | None = None,
    min_size: int | None = None,
    eps: float = 1.6,
    return_graph: bool = False,
    kmeans: bool = True,
) -> tuple[list[Patch], nx.Graph]:
    """Create patches for points. Starts by sampling patch centers and
    assigning points to the nearest center and any center that is within `eps`
    of the nearest center to create patches. Patches are then grown by incrementally
    adding the next closest point until the patch degree constraint is satisfied.
    Finally, patches that are smaller than `min_size` are expanded, and shortest edges
    are added to make the patch graph connected if necessary.

    Args:
        points (np.ndarray): ndarray of floats of shape (N, d), d-dimensional embedding.

        sample_size: Number of patches splitting the set of N points,
            default is 100.

        min_degree: Minimum patch degree, defaults to `d + 1`.

        min_overlap: Minimum overlap to consider two patches connected,
            default is `d + 1`.

        min_size: Minimum patch size, defaults to `len(points) / sample_size`,
            default is None.

        eps: Tolerance for expanding initial Voronoi patches, default is 1.6.

        return_graph: If True, returns the patch graph as a networkx Graph,
            default is False.

        kmeans: If True, chooses patch centers using k-means,
            otherwise, patch centers are sampled uniformly at random from points, default is True.

    Returns:
        List of patches.

        Patch graph if `return_graph=True`.
    """
    n, d = points.shape
    if min_size is None:
        min_size = n // sample_size
    if min_degree is None:
        min_degree = d + 1
    if min_overlap is None:
        min_overlap = d + 1

    # Find patch centers
    if kmeans:
        k_means = KMeans(n_clusters=sample_size, random_state=ut.rg.integers(2**32 - 1))
        k_means.fit(points)
        centers = k_means.cluster_centers_
    else:
        sample_mask = ut.rg.choice(len(points), size=sample_size, replace=False)
        centers = points[sample_mask, :]

    # list of node indeces for each patch
    node_lists = [[] for _ in centers]
    patch_index = [[] for _ in range(n)]
    overlaps = [Counter() for _ in range(sample_size)]

    # compute distance to centers
    distances = cdist(centers, points)

    # build eps-Voronoi patches
    index = np.argsort(distances, axis=0)
    for node in range(n):
        patch = index[0, node]
        node_lists[patch].append(node)
        for other in patch_index[node]:
            overlaps[other][patch] += 1
            overlaps[patch][other] += 1
        patch_index[node].append(patch)
        min_dist = distances[patch, node]
        for patch in index[1:, node]:
            if distances[patch, node] < eps * min_dist:
                node_lists[patch].append(node)
                for other in patch_index[node]:
                    overlaps[other][patch] += 1
                    overlaps[patch][other] += 1
                patch_index[node].append(patch)
            else:
                break

    # grow patches until degree constraints and size constraints are satisfied

    # find patches that do not satisfy the constraints
    grow = {
        i
        for i, ov in enumerate(overlaps)
        if len(node_lists[i]) < min_size
        or sum(v >= min_overlap for v in ov.values()) < min_degree
    }

    # sort distance matrix (make sure patch members are sorted first)
    for i, nodes in enumerate(node_lists):
        distances[i, nodes] = -1
    index = np.argsort(distances, axis=1)

    while grow:
        patches = list(grow)
        for patch in patches:
            size = len(node_lists[patch])
            if size >= n or (
                size >= min_size
                and sum(v >= min_overlap for v in overlaps[patch].values())
                >= min_degree
            ):
                grow.remove(patch)
            else:
                next_node = index[patch, size]
                node_lists[patch].append(next_node)
                for other in patch_index[next_node]:
                    overlaps[other][patch] += 1
                    overlaps[patch][other] += 1
                patch_index[next_node].append(patch)

    # check patch network is connected and add edges if necessary
    patch_network = nx.Graph()
    for i, others in enumerate(overlaps):
        for other, ov in others.items():
            if ov >= min_overlap:
                patch_network.add_edge(i, other)

    if not nx.is_connected(patch_network):
        components = list(nx.connected_components(patch_network))
        edges = []
        for c1, patches1 in enumerate(components):
            patches1 = list(patches1)
            for it, patches2 in enumerate(components[c1 + 1 :]):
                patches2 = list(patches2)
                c2 = c1 + it + 1
                patch_distances = cdist(centers[patches1, :], centers[patches2, :])

                # TODO: fix possible unbalanced-tuple-unpacking
                # pylint: disable=unbalanced-tuple-unpacking
                i, j = np.unravel_index(
                    np.argmin(patch_distances), patch_distances.shape
                )
                # pylint: enable=unbalanced-tuple-unpacking

                edges.append((patch_distances[i, j], patches1[i], patches2[j], c1, c2))
        edges.sort()
        component_graph = nx.Graph()
        component_graph.add_nodes_from(range(len(components)))
        for _, i, j, c1, c2 in edges:
            nodes1 = set(node_lists[i])
            nodes2 = set(node_lists[j])
            nodes = nodes1.union(nodes2)
            dist_list = [
                (distances[i, node] + distances[j, node], node) for node in nodes
            ]
            dist_list.sort()
            for it in range(min_overlap):
                node = dist_list[it][1]
                if node not in nodes1:
                    node_lists[i].append(node)
                if node not in nodes2:
                    node_lists[j].append(node)
            component_graph.add_edge(c1, c2)
            patch_network.add_edge(i, j)
            if nx.is_connected(component_graph):
                break

    if return_graph:
        return [Patch(nodes, points[nodes, :]) for nodes in node_lists], patch_network

    return [Patch(nodes, points[nodes, :]) for nodes in node_lists]


# pylint: enable=too-many-statements
# pylint: enable=too-many-branches


def rand_scale_patches(
    alignment_problem: ut.AlignmentProblem, min_scale: float = 1e-2
) -> list[float]:
    """
    randomly scale patches of alignment problem and return the true scales (used for testing)

    Args:
        alignment_problem (AlignmentProblem): Alignment problem to be rescaled.

        min_scale: Minimum scale factor (scale factors are sampled
                           log-uniformly from the interval [min_scale, 1/min_scale]),
                           default is 1e-2.

    Returns:
        List of true scales.
    """

    scales = np.exp(
        ut.rg.uniform(
            np.log(min_scale), np.log(1 / min_scale), alignment_problem.n_patches
        )
    )
    alignment_problem.scale_patches(scales)
    return scales


def rand_rotate_patches(alignment_problem: ut.AlignmentProblem) -> list[np.ndarray]:
    """Randomly rotate patches of alignment problem and return true rotations (used for testing)

    Args:
        alignment_problem: Alignment problem to be transformed.

    Returns:
        List of true rotations.
    """

    rotations = [rand_orth(alignment_problem.dim) for _ in alignment_problem.patches]
    alignment_problem.rotate_patches(rotations)
    return rotations


def rand_shift_patches(
    alignment_problem: ut.AlignmentProblem, shift_scale: float = 100.0
) -> np.ndarray:
    """Randomly shift patches by adding a normally distributed vector (used for testing)

    Args:
        alignment_problem (AlignmentProblem): Alignment problem to be transformed.

        shift_scale: Standard deviation for shifts, default is 100.0.

    Returns:
        np.ndarray: Array of true shifts.
    """
    shifts = ut.rg.normal(
        loc=0,
        scale=shift_scale,
        size=(alignment_problem.n_patches, alignment_problem.dim),
    )
    alignment_problem.translate_patches(shifts)
    return shifts


def add_noise(
    alignment_problem: ut.AlignmentProblem,
    noise_level: float = 1,
    scales: np.ndarray | None = None,
):
    """Add random normally-distributed noise to each point in each patch.

    Args:
        alignment_problem (AlignmentProblem): The alignment problem to be transformed.

        noise_level: Standard deviation of noise, default is 1.

        scales: Array of scales for each patch, default is None.
            Noise for each patch is multiplied by the corresponding scale.

    """
    if noise_level > 0:
        if scales is None:
            scales = np.ones(alignment_problem.n_patches)
        for patch, scale in zip(alignment_problem.patches, scales):
            noise = ut.rg.normal(loc=0, scale=noise_level * scale, size=patch.shape)
            patch.coordinates += noise


def noise_profile(
    points: np.ndarray,
    base_problem: ut.AlignmentProblem,
    max_noise: float = 0.5,
    steps: int = 101,
    scales: np.ndarray | None = None,
    types: list[Type[ut.AlignmentProblem]] | None = None,
    labels: list[str] | None = None,
    min_overlap: list[int] | None = None,
    plot: bool | None = True,
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Plot procrustes reconstruction errors as a function of the noise level.

    Args:
        points (np.ndarray): True data.

        base_problem (AlignmentProblem): Alignment problem without noise
            (usually should have rotated/shifted/scaled patches).

        max_noise: Maximum standard deviation for noise, default is 0.5.

        steps: Number of noise steps between 0 and `max_noise`, default is 101.

        scales: Scales of patches (noise is scaled accordingly),
            default is None.

        types: List of AlignmentProblem subclasses to test
            (each is tested with the same noise), default is None.

        labels: Labels to use for the legend, default is None.

        min_overlap: Values of `min_overlap` to include in the test,
            default is None.

        plot: Plot results, default is True.

    Returns:
        noise_levels, errors.
    """

    # set up labels and min_overlap
    if min_overlap is not None:
        if not isinstance(min_overlap, Iterable):
            min_overlap = [min_overlap]
        else:
            if labels is not None:
                labels = [f"{label}-{ov}" for label in labels for ov in min_overlap]

    if types is None:
        types = [ut.AlignmentProblem]
    errors = [[] for _ in range(steps)]
    noise_levels = np.linspace(0, max_noise, steps)
    for e_l, noise in zip(errors, noise_levels):
        noisy_problem = copy(base_problem)
        add_noise(noisy_problem, noise, scales)
        for problem_cls in types:
            if min_overlap is None:
                problem = copy(noisy_problem)
                problem.__class__ = problem_cls
                e_l.append(
                    ut.procrustes_error(
                        points, problem.get_aligned_embedding(scale=True)
                    )
                )
            else:
                for ov in min_overlap:
                    ov_problem = problem_cls(noisy_problem.patches, min_overlap=ov)
                    e_l.append(
                        ut.procrustes_error(
                            points, ov_problem.get_aligned_embedding(scale=True)
                        )
                    )
        print(f"Noise: {noise}, errors: {e_l}")

    if plot:
        plt.plot(noise_levels, errors)
        if labels is not None:
            plt.legend(labels)
        plt.xlabel("noise level")
        plt.ylabel("procrustes errors")

    return noise_levels, errors


def plot_reconstruction(
    points: np.ndarray, problem: ut.AlignmentProblem, scale: bool = True
) -> float:
    """Plot the reconstruction error for each point.

    Args:
        points: True positions.

        problem: Alignment problem.

        scale: Rescale patches, default is True.

    Returns:
        float: Reconstruction error.
    """
    recovered_pos = problem.get_aligned_embedding(scale=scale)
    points, recovered_pos, error = procrustes(points, recovered_pos)
    plt.plot(
        np.array([points[:, 0], recovered_pos[:, 0]]),
        np.array([points[:, 1], recovered_pos[:, 1]]),
        "k",
        linewidth=0.5,
    )
    plt.plot(recovered_pos[:, 0], recovered_pos[:, 1], "k.", markersize=1)
    for patch in problem.patches:
        index = list(patch.index.keys())
        old_c = np.mean(points[index, :], axis=0)
        new_c = np.mean(recovered_pos[index, :], axis=0)
        plt.plot([old_c[0], new_c[0]], [old_c[1], new_c[1]], "r", linewidth=1)
        plt.plot([new_c[0]], [new_c[1]], "r.", markersize=2)
    return error


def save_data(points: np.ndarray, filename: str):
    """Save an array of points to a CSV file.

    Ensures the specified filename has a `.csv` extension and
    writes the data to a CSV file with UTF-8 encoding.

    Args:
        points (np.ndarray): Array of data points to save, where each row is a data point.
        filename (str): Desired filename for saving the data.

    """
    filename = ut.ensure_extension(filename, ".csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(points)


def rand_orth(dim: int) -> np.ndarray:
    """
    Sample a random orthogonal matrix (for testing)
    Use normal distribution to ensure uniformity.

    Args:
        dim (int): The dimension of the orthogonal matrix.

    Returns:
        np.ndarray: A randomly generated orthogonal matrix of shape `(dim, dim)`.
    """

    a = ut.rg.normal(size=(dim, 1))
    a = a / np.sqrt(a.T.dot(a))
    m = a

    for _ in range(dim - 1):
        a = ut.rg.normal(size=(dim, 1))
        a = a - m.dot(m.T).dot(a)
        a = a / np.sqrt(a.T.dot(a))
        m = np.hstack((m, a))
    return m


def main(arguments: argparse.Namespace):
    """Generate synthetic data and test the alignment algorithms.

    Args:

        arguments: argparse.Namespace object
    """

    ut.random_gen(arguments.seed)
    problem_types = [
        ut.AlignmentProblem,
        ut.WeightedAlignmentProblem,
        ut.SVDAlignmentProblem,
    ]
    labels = ["standard", "weighted", "svd-weighted"]

    # generate random data
    points = generate_data(
        n_clusters=arguments.n_clusters,
        scale=arguments.max_shift,
        std=arguments.max_var,
        max_size=arguments.max_size,
        dim=arguments.dim,
    )
    outdir = Path(arguments.outdir)
    save_data(points, filename=outdir / "points.csv")
    patches = voronoi_patches(
        points=points,
        sample_size=arguments.sample_size,
        min_degree=arguments.min_degree,
        min_overlap=arguments.min_overlap,
        min_size=arguments.min_size,
        eps=arguments.eps,
        kmeans=arguments.kmeans,
    )
    base_problem = ut.AlignmentProblem(patches, min_overlap=arguments.min_overlap)
    rand_shift_patches(base_problem)
    scales = rand_scale_patches(base_problem)
    rand_rotate_patches(base_problem)

    print(f"Mean patch degree: {mean(base_problem.patch_degrees)}")
    if arguments.steps > 0:
        plt.figure()
        noise_profile(
            points,
            base_problem,
            steps=arguments.steps,
            max_noise=arguments.max_noise,
            scales=scales,
            types=problem_types,
            labels=labels,
            min_overlap=arguments.min_recovery_overlap,
        )
        plt.savefig(path.join(arguments.outdir, "noise_profile.pdf"))
        plt.close()

    for noise in arguments.plot_noise:
        if arguments.dim > 2:
            raise RuntimeError("plotting reconstruction error only works for dim=2")
        noisy_problem = copy(base_problem)
        add_noise(noisy_problem, noise_level=noise, scales=scales)
        for problem_cls, label in zip(problem_types, labels):
            plt.figure()
            problem = copy(noisy_problem)
            problem.__class__ = problem_cls
            error = plot_reconstruction(points, problem)
            plt.title(f"Noise: {noise}, error: {error}")
            plt.savefig(
                path.join(arguments.outdir, f"errorplot_{label}_noise{noise}.pdf")
            )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run local2global example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_clusters", default=5, type=int, help="Number of clusters in test data"
    )
    parser.add_argument("--max_shift", default=1, type=float, help="Cluster shift")
    parser.add_argument(
        "--kmeans", action="store_true", help="use kmeans to find patch centers"
    )
    parser.add_argument("--max_var", default=0.2, type=float, help="Cluster dispersion")
    parser.add_argument("--max_size", default=2000, type=int, help="Max cluster size")
    parser.add_argument("--sample_size", default=10, type=int, help="Number of patches")
    parser.add_argument("--dim", default=2, type=int, help="Data dimension")
    parser.add_argument(
        "--eps", default=1.6, type=float, help="Tolerance for patch overlaps"
    )
    parser.add_argument(
        "--min_overlap",
        type=int,
        default=10,
        help="Minimum patch overlap for connectivity constraint",
    )
    parser.add_argument(
        "--min_recovery_overlap",
        type=int,
        default=[],
        action="append",
        help="Minimum patch overlap for recovery (defaults to min_overlap)",
    )
    parser.add_argument("--min_size", type=int, default=10, help="Minimum patch size")
    parser.add_argument(
        "--min_degree", type=int, default=None, help="Minimum patch degree"
    )
    parser.add_argument(
        "--max_noise", default=0.3, type=float, help="Maximum noise level"
    )
    parser.add_argument(
        "--steps", default=101, type=int, help="Number of steps for noise profile"
    )
    parser.add_argument(
        "--plot_noise",
        "-p",
        default=[],
        action="append",
        type=float,
        help="Noise level to plot (can be specified multiple times)",
    )
    parser.add_argument("--outdir", "-o", type=str, help="output dir", default=".")
    parser.add_argument("--seed", default=None, type=int, help="Seed for rng")
    args = parser.parse_args()

    if not args.min_recovery_overlap:
        args.min_recovery_overlap = None

    main(args)
