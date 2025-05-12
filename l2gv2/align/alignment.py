"""
Base functions and classes for alignment problems.
"""

import scipy as sp
import copy
import json
from collections import defaultdict
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial import procrustes

# local imports
from l2gv2.patch import Patch
from l2gv2.utils import ensure_extension


def procrustes_error(coordinates1, coordinates2):
    """
    compute the procrustes alignment error between two sets of coordinates

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    return procrustes(coordinates1, coordinates2)[2]


def local_error(patch: Patch, reference_coordinates):
    """
    compute the euclidean distance between patch coordinate and reference
    coordinate for each node in patch

    Args:
        patch:
        reference_coordinates:

    Returns:
        vector of error values
    """
    return np.linalg.norm(
        reference_coordinates[patch.nodes, :] - patch.coordinates, axis=1
    )


def transform_error(transforms):
    """
    Compute the recovery error based on tracked transformations.

    After recovery, all transformations should be constant across patches
    as we can recover the embedding only up to a global scaling/rotation/translation.
    The error is computed as the mean over transformation elements of the standard deviation over patches.

    Args:
        transforms: list of transforms
    """
    return np.mean(np.std(transforms, axis=0))


def orthogonal_MSE_error(rots1, rots2):
    """
    Compute the MSE between two sets of orthogonal transformations up to a global transformation

    Args:
        rots1: First list of orthogonal matrices
        rots2: Second list of orthogonal matrices

    """
    dim = len(rots1[0])
    rots1 = np.asarray(rots1)
    rots1 = rots1.transpose((0, 2, 1))
    rots2 = np.asarray(rots2)
    combined = np.mean(rots1 @ rots2, axis=0)
    _, s, _ = sp.linalg.svd(combined)
    return 2 * (dim - np.sum(s))


class AlignmentProblem:  # pylint: disable=too-many-instance-attributes
    """
    Base class for alignment problems.
    """

    n_nodes = None
    n_patches = None
    dim = None

    scales = None
    rotations = None
    shifts = None

    verbose = False

    def weight(self, _i, _j):
        """
        Compute the weight for a pair of patches
        """
        # pylint: disable=unused-argument
        return 1

    def __init__(
        self,
        patches: list[Patch],
        patch_edges=None,
        min_overlap=None,
        copy_data=True,
        self_loops=False,
        verbose=False,
    ):
        """
        Initialise the alignment problem with a list of patches

        Args:
            patches: List of patches to synchronise
            patch_edges: if provided, only compute relative transformations for given patch edges (all pairs of patches
                         with at least ``min_overlap`` points in common are included by default)
            min_overlap (int): minimum number of points in the overlap required for two patches to be considered
                               connected (defaults to `dim+1`) where `dim` is the embedding dimension of the patches
            copy_data (bool): if ``True``, input patches are copied (default: ``True``)
            self_loops (bool): if ``True``, self-loops from a patch to itself are included in the synchronisation problem
                               (default: ``False``)
            verbose(bool): if True print diagnostic information (default: ``False``)

        """
        if copy_data:
            self.patches = [copy.copy(patch) for patch in patches]
        else:
            self.patches = patches
        self.verbose = verbose
        self.n_nodes = max(max(patch.index.keys()) for patch in self.patches) + 1
        self.n_patches = len(self.patches)
        self.dim = self.patches[0].shape[1]

        self.scales = np.ones(self.n_patches)
        self.rotations = np.tile(np.eye(self.dim), (self.n_patches, 1, 1))
        self.shifts = np.zeros((self.n_patches, self.dim))
        self._aligned_embedding = None

        if min_overlap is None:
            min_overlap = self.dim + 1

        # create an index for the patch membership of each node
        self.patch_index = [[] for _ in range(self.n_nodes)]
        for i, patch in enumerate(self.patches):
            for node in patch.nodes:
                self.patch_index[node].append(i)

        # find patch overlaps
        self.patch_overlap = defaultdict(list)
        for i, patch in enumerate(self.patches):
            for node in patch.index:
                for j in self.patch_index[node]:
                    if self_loops or i != j:
                        self.patch_overlap[i, j].append(node)

        # restrict to patch edges if provided
        if patch_edges is not None:
            self.patch_overlap = {e: self.patch_overlap[e] for e in patch_edges}

        # remove small overlaps
        keys = list(self.patch_overlap.keys())
        for e in keys:
            if self_loops or e[0] != e[1]:
                if len(self.patch_overlap[e]) < min_overlap:
                    if patch_edges is None:
                        del self.patch_overlap[e]
                    else:
                        raise RuntimeError("Patch edges do not satisfy minimum overlap")
            else:
                del self.patch_overlap[e]  # remove spurious self-loops

        # find patch degrees
        self.patch_degrees = [0] * self.n_patches
        for i, j in self.patch_overlap.keys():
            self.patch_degrees[i] += 1

        patch_graph = nx.Graph()
        patch_graph.add_edges_from(self.patch_overlap.keys())
        if nx.number_connected_components(patch_graph) > 1:
            raise RuntimeError("patch graph is not connected")

        if self.verbose:
            print(f"mean patch degree: {np.mean(self.patch_degrees)}")

    def mean_embedding(self, out=None):
        """
        Compute node embeddings as the centroid over patch embeddings

        Args:
            out: numpy array to write results to (supply a memmap for large-scale problems that do not fit in ram)
        """
        if out is None:
            embedding = np.zeros((self.n_nodes, self.dim))
        else:
            embedding = out  # important: needs to be zero-initialised

        count = np.array([len(patch_list) for patch_list in self.patch_index])
        for patch in tqdm(
            self.patches,
            smoothing=0,
            desc="Compute mean embedding",
            disable=self.verbose,
        ):
            embedding[patch.nodes] += patch.coordinates

        embedding /= count[:, None]

        return embedding

    def median_embedding(self, out=None):
        if out is None:
            out = np.full((self.n_nodes, self.dim), np.nan)

        for i, pids in tqdm(
            enumerate(self.patch_index),
            total=self.n_nodes,
            desc="Compute median embedding for node",
            disable=self.verbose,
        ):
            if pids:
                points = np.array([self.patches[pid].get_coordinate(i) for pid in pids])
                out[i] = np.median(points, axis=0)
        return out

    def scale_patches(self, scale_factors: np.ndarray):
        """
        Synchronise scales of the embeddings for each patch

        Args:
            scale_factors: if provided apply the given scales instead of synchronising
        """
        for i, scale in enumerate(scale_factors):
            self.patches[i].coordinates *= scale
            # track transformations
            self.scales[i] *= scale
            self.shifts[i] *= scale
        return self

    def rotate_patches(self, rotations: np.ndarray):
        """align the rotation/reflection of all patches

        Args:
            rotations: If provided, apply the given transformations instead of synchronizing patch rotations
        """
        for i, rot in enumerate(rotations):
            self.patches[i].coordinates = self.patches[i].coordinates @ rot.T
            # track transformations
            self.rotations[i] = self.rotations[i] @ rot.T
            self.shifts[i] = self.shifts[i] @ rot.T
        return self

    def translate_patches(self, translations: np.ndarray):
        """align the patches by translation

        Args:
            translations: If provided, apply the given translations instead of synchronizing

        """
        for i, t in enumerate(translations):
            self.patches[i].coordinates += t
            # keep track of transformations
            self.shifts[i] += t
        return self

    def align_patches(self, _scale=False):
        """
        Align the patches. This is where the magic happens.
        """
        raise NotImplementedError("Align patches is not implemented for this aligner")

    def get_aligned_embedding(self, scale=False, realign=False, out=None):
        """Return the aligned embedding

        Args:
            scale (bool): if ``True``, rescale patches (default: ``False``)
            realign (bool): if ``True``, recompute aligned embedding even if it already exists (default: ``False``)

        Returns:
            n_nodes x dim numpy array of embedding coordinates
        """
        if realign or self._aligned_embedding is None:
            self._aligned_embedding = self.align_patches(scale).mean_embedding(out)
        return self._aligned_embedding

    def save_patches(self, filename):
        """
        save patch embeddings to json file
        Args:
            filename: path to output file


        """
        filename = ensure_extension(filename, ".json")
        patch_dict = {
            str(i): {
                int(node): [float(c) for c in coord]
                for node, coord in zip(patch.index, patch.coordinates)
            }
            for i, patch in enumerate(self.patches)
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(patch_dict, f)

    @classmethod
    def load(cls, filename):
        """
        restore ``AlignmentProblem`` from patch file

        Args:
            filename: path to patch file

        """
        filename = ensure_extension(filename, ".json")
        with open(filename, encoding="utf-8") as f:
            patch_dict = json.load(f)
        patch_list = [None] * len(patch_dict)
        for i, patch_data in patch_dict.items():
            nodes = (int(n) for n in patch_data.keys())
            coordinates = list(patch_data.values())
            patch_list[int(i)] = Patch(nodes, coordinates)
        return cls(patch_list)

    def save_embedding(self, filename):
        """
        save aligned embedding to json file

        Args:
            filename: output filename

        """
        filename = ensure_extension(filename, ".json")
        embedding = {str(i): c for i, c in enumerate(self.get_aligned_embedding())}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(embedding, f)

    def __copy__(self):
        """return a copy of the alignment problem where all patches are copied."""
        instance = type(self).__new__(type(self))
        for key, value in self.__dict__.items():
            instance.__dict__[key] = copy.copy(value)
        instance.patches = [copy.copy(patch) for patch in self.patches]
        return instance


# See https://github.com/OxfordRSE/L2Gv2/issues/45 for abstract method inheritance discussion
class WeightedAlignmentProblem(AlignmentProblem):  # pylint: disable=abstract-method
    """
    Variant of the local2global algorithm where patch edges are weighted according to the number of nodes in the overlap.
    """

    def weight(self, i, j):
        """
        compute weight for pair of patches

        Args:
            i: first patch index
            j: second patch index

        Returns:
            number of shared nodes between patches `i` and `j`
        """
        ov = len(self.patch_overlap[i, j])
        return ov
