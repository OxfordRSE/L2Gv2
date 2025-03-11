#  Copyright (c) 2021. Lucas G. S. Jeub
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import numpy as np
import scipy as sp
import scipy.sparse as ss
from scipy.sparse.linalg import lsmr
from tqdm.auto import tqdm
import json
import copy
from typing import Callable, Any

from l2gv2.patch import Patch
from l2gv2.utils import ensure_extension
from l2gv2.align.registry import register_aligner
from l2gv2.align.alignment import AlignmentProblem

rg = np.random.default_rng()


def _cov_svd(coordinates1: np.ndarray, coordinates2: np.ndarray):
    """
    Compute SVD of covariance matrix between two sets of coordinates

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    coordinates1 = coordinates1 - coordinates1.mean(axis=0)
    coordinates2 = coordinates2 - coordinates2.mean(axis=0)
    cov = coordinates1.T @ coordinates2
    return sp.linalg.svd(cov)


def relative_orthogonal_transform(coordinates1, coordinates2):
    """
    Find the best orthogonal transformation aligning two sets of coordinates for the same nodes

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    # Note this is completely equivalent to the approach in
    # "Closed-Form Solution of Absolute Orientation using Orthonormal Matrices"
    # Journal of the Optical Society of America A · July 1988
    U, _, Vh = _cov_svd(coordinates1, coordinates2)
    return U @ Vh


def nearest_orthogonal(mat):
    """
    Compute nearest orthogonal matrix to a given input matrix

    Args:
        mat: input matrix
    """
    U, _, Vh = sp.linalg.svd(mat)
    return U @ Vh


def relative_scale(coordinates1, coordinates2, clamp=1e8):
    """
    compute relative scale of two sets of coordinates for the same nodes

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    scale1 = np.linalg.norm(coordinates1 - np.mean(coordinates1, axis=0))
    scale2 = np.linalg.norm(coordinates2 - np.mean(coordinates2, axis=0))
    if scale1 > clamp * scale2:
        print("extremely large scale clamped")
        return clamp
    if scale1 * clamp < scale2:
        print("extremely small scale clamped")
        return 1 / clamp
    return scale1 / scale2


@register_aligner("l2g")
class L2GAlignmentProblem(AlignmentProblem):
    """
    Implements the standard local2global algorithm using an unweighted patch graph
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
        super().__init__(
            patches=patches,
            patch_edges=patch_edges,
            min_overlap=min_overlap,
            copy_data=copy_data,
            self_loops=self_loops,
            verbose=verbose,
        )

    def scale_patches(self, scale_factors=None):
        """
        Synchronise scales of the embeddings for each patch

        Args:
            scale_factors: if provided apply the given scales instead of synchronising
        """
        if scale_factors is None:
            scale_factors = [1 / x for x in self.calc_synchronised_scales()]

        for i, scale in enumerate(scale_factors):
            self.patches[i].coordinates *= scale
            # track transformations
            self.scales[i] *= scale
            self.shifts[i] *= scale
        return self

    def calc_synchronised_scales(self, max_scale=1e8):
        """
        Compute the scaling transformations that best align the patches

        Args:
            max_scale: maximum allowed scale (all scales are clipped to the range [``1/max_scale``, ``max_scale``])
                       (default: 1e8)

        Returns:
            list of scales

        """
        scaling_mat = self._transform_matrix(
            lambda ov1, ov2: relative_scale(ov1, ov2, max_scale), 1
        )
        vec = self._synchronise(scaling_mat, 1)
        vec = vec.flatten()
        vec = np.abs(vec)
        vec /= vec.mean()
        vec = np.clip(
            vec, a_min=1 / max_scale, a_max=max_scale, out=vec
        )  # avoid blow-up
        return vec

    def rotate_patches(self, rotations=None):
        """align the rotation/reflection of all patches

        Args:
            rotations: If provided, apply the given transformations instead of synchronizing patch rotations
        """
        if rotations is None:
            rotations = (rot.T for rot in self.calc_synchronised_rotations())

        for i, rot in enumerate(rotations):
            self.patches[i].coordinates = self.patches[i].coordinates @ rot.T
            # track transformations
            self.rotations[i] = self.rotations[i] @ rot.T
            self.shifts[i] = self.shifts[i] @ rot.T
        return self

    def calc_synchronised_rotations(self):
        """Compute the orthogonal transformations that best align the patches"""
        rots = self._transform_matrix(
            relative_orthogonal_transform, self.dim, symmetric_weights=True
        )
        vecs = self._synchronise(rots, blocksize=self.dim, symmetric=True)
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)
        return vecs

    def translate_patches(self, translations=None):
        """align the patches by translation

        Args:
            translations: If provided, apply the given translations instead of synchronizing

        """
        if translations is None:
            translations = self.calc_synchronised_translations()

        for i, t in enumerate(translations):
            self.patches[i].coordinates += t
            # keep track of transformations
            self.shifts[i] += t
        return self

    def calc_synchronised_translations(self):
        """Compute translations that best align the patches"""
        b = np.empty((len(self.patch_overlap), self.dim))
        row = []
        col = []
        val = []
        for i, ((p1, p2), overlap) in enumerate(self.patch_overlap.items()):
            row.append(i)
            col.append(p1)
            val.append(-1)
            row.append(i)
            col.append(p2)
            val.append(1)
            b[i, :] = np.mean(
                self.patches[p1].get_coordinates(overlap)
                - self.patches[p2].get_coordinates(overlap),
                axis=0,
            )
        A = ss.coo_matrix(
            (val, (row, col)),
            shape=(len(self.patch_overlap), self.n_patches),
            dtype=np.int8,
        )
        A = A.tocsr()
        translations = np.empty((self.n_patches, self.dim))
        for d in range(self.dim):
            translations[:, d] = lsmr(A, b[:, d], atol=1e-16, btol=1e-16)[0]
            # TODO: probably doesn't need to be that accurate, this is for testing
        return translations

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

    def align_patches(self, scale=False):
        if scale:
            self.scale_patches()
        self.rotate_patches()
        self.translate_patches()
        return self

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

    def _synchronise(self, matrix: ss.spmatrix, blocksize=1, symmetric=False):
        dim = matrix.shape[0]
        if symmetric:
            matrix = matrix + ss.eye(
                dim
            )  # shift to ensure matrix is positive semi-definite for buckling mode
            eigs, vecs = ss.linalg.eigsh(
                matrix,
                k=blocksize,
                v0=rg.normal(size=dim),
                which="LM",
                sigma=2,
                mode="buckling",
            )
            # eigsh unreliable with multiple (clustered) eigenvalues, only buckling mode seems to help reliably

        else:
            # scaling is not symmetric but Perron-Frobenius applies
            eigs, vecs = ss.linalg.eigs(matrix, k=blocksize, v0=rg.normal(size=dim))
            eigs = eigs.real
            vecs = vecs.real

        order = np.argsort(eigs)
        vecs = vecs[:, order[-1 : -blocksize - 1 : -1]]
        if self.verbose:
            print(f"eigenvalues: {eigs}")
        vecs.shape = (dim // blocksize, blocksize, blocksize)
        return vecs

    def _transform_matrix(
        self,
        transform: Callable[[np.ndarray, np.ndarray], Any],
        dim,
        symmetric_weights=False,
    ):
        """Calculate matrix of relative transformations between patches

        Args:
            transform: function to compute the relative transformation
            dim: output dimension of transform should be `(dim, dim)`
            symmetric_weights: if true use symmetric weighting (default: False)
        """
        n = self.n_patches  # number of patches
        if dim != 1:
            # construct matrix of rotations as a block-sparse-row matrix
            data = np.empty(shape=(len(self.patch_overlap), dim, dim))
        else:
            data = np.empty(shape=(len(self.patch_overlap),))
        weights = np.zeros(n)
        indptr = np.zeros((n + 1,), dtype=int)
        np.cumsum(self.patch_degrees, out=indptr[1:])
        index = np.empty(shape=(len(self.patch_overlap),), dtype=int)

        keys = sorted(self.patch_overlap.keys())
        # TODO: this could be sped up by a factor of two by not computing rotations twice
        for count, (i, j) in tqdm(
            enumerate(keys),
            total=len(keys),
            desc="Compute relative transformations",
            disable=self.verbose,
        ):
            if i == j:
                element = np.eye(dim)
            else:
                overlap_idxs = self.patch_overlap[i, j]
                # find positions of overlapping nodes in the two reference frames
                overlap1 = self.patches[i].get_coordinates(overlap_idxs)
                overlap2 = self.patches[j].get_coordinates(overlap_idxs)
                element = transform(overlap1, overlap2)
            weight = self.weight(i, j)
            weights[i] += weight
            element *= weight
            data[count] = element

        # computed weighted average based on error weights
        if symmetric_weights:
            for i in range(n):
                for ind in range(indptr[i], indptr[i + 1]):
                    data[ind] /= np.sqrt(weights[i] * weights[index[ind]])
        else:
            for i in range(n):
                data[indptr[i] : indptr[i + 1]] /= weights[i]
        if dim == 1:
            matrix = ss.csr_matrix((data, index, indptr), shape=(n, n))
        else:
            matrix = ss.bsr_matrix(
                (data, index, indptr), shape=(dim * n, dim * n), blocksize=(dim, dim)
            )
        return matrix
