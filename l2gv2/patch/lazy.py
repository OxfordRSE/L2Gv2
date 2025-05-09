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

"""
Lazy coordinates for patch embeddings
"""

import copy
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm


class BaseLazyCoordinates(ABC):
    """TODO: docstring for `BaseLazyCoordinates`"""

    @property
    @abstractmethod
    def shape(self):
        """Shape of the coordinates"""
        raise NotImplementedError

    def __array__(self, dtype=None):
        return np.asanyarray(self[:], dtype=dtype)

    @abstractmethod
    def __iadd__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        new = copy.copy(self)
        new += other
        return new

    @abstractmethod
    def __isub__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        new = copy.copy(self)
        new -= other
        return new

    @abstractmethod
    def __imul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        new = copy.copy(self)
        new *= other
        return new

    @abstractmethod
    def __itruediv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        new = copy.copy(self)
        new /= other
        return new

    @abstractmethod
    def __imatmul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        new = copy.copy(self)
        new @= other
        return new

    @abstractmethod
    def __getitem__(self, item) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]


class LazyCoordinates(BaseLazyCoordinates):
    """TODO: doctring for `LazyCoordinates`"""

    def __init__(self, x, shift=None, scale=None, rot=None):
        self._x = x
        dim = self.shape[1]
        if shift is None:
            self.shift = np.zeros((1, dim))
        else:
            self.shift = np.array(shift).reshape((1, dim))

        if scale is None:
            self.scale = 1
        else:
            self.scale = scale

        if rot is None:
            self.rot = np.eye(dim)
        else:
            self.rot = np.array(rot)

    def save_transform(self, filename):
        """Save the transformation to a file

        Args:

            filename: filename to save the transformation to
        """

        np.savez(filename, shift=self.shift, scale=self.scale, rot=self.rot)

    @property
    def shape(self):
        return self._x.shape

    def __copy__(self):
        new = self.__class__(self._x, self.shift, self.scale, self.rot)
        for key, value in self.__dict__.items():
            if key not in new.__dict__:
                new.__dict__[key] = value
        return new

    def __iadd__(self, other):
        self.shift += other
        return self

    def __isub__(self, other):
        self.shift -= other
        return self

    def __imul__(self, other):
        self.scale *= other
        self.shift *= other
        return self

    def __itruediv__(self, other):
        self.scale /= other
        self.shift /= other
        return self

    def __imatmul__(self, other):
        self.rot = self.rot @ other
        self.shift = self.shift @ other
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            x = self._x[item[0]]
        else:
            x = self._x[item]
        x = x * self.scale
        x = x @ self.rot
        x += self.shift
        if isinstance(item, tuple):
            if x.ndim > 1:
                return x[(slice(None), *item[1:])]
            return x[item[1]]

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._x)})"


class LazyFileCoordinates(LazyCoordinates):
    """TODO: doctring for `LazyFileCoordinates`"""

    def __init__(self, filename, *args, **kwargs):
        with open(filename, "rb") as f:
            # TODO: review this, commented out because the output is not used
            # major, minor = np.lib.format.read_magic(f)

            shape, *_ = np.lib.format.read_array_header_1_0(f)
        self._shape = shape
        super().__init__(filename, *args, **kwargs)

    @property
    def _x(self):
        return np.load(self.filename, mmap_mode="r")

    @_x.setter
    def _x(self, other):
        self.filename = other

    @property
    def shape(self):
        return self._shape

    def __copy__(self):
        new = self.__class__(self.filename, self.shift, self.scale, self.rot)
        for key, value in self.__dict__.items():
            if key not in new.__dict__:
                new.__dict__[key] = value
        return new


class LazyMeanAggregatorCoordinates(BaseLazyCoordinates):
    """TODO: doctring for `LazyMeanAggregatorCoordinates`"""

    def __init__(self, patches):
        self.patches = []
        for patch in patches:
            if isinstance(patch.coordinates, LazyMeanAggregatorCoordinates):
                # flatten hierarchy
                self.patches.extend(patch.coordinates.patches)
            else:
                self.patches.append(patch)
        self.dim = patches[0].shape[1]
        nodes = set()
        for patch in patches:
            nodes.update(patch.nodes)

        self.nodes = np.array(sorted(nodes))

    @property
    def shape(self):
        return len(self.nodes), self.dim

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item, *others = item
        else:
            others = ()
        nodes = self.nodes[item]
        out = self.get_coordinates(nodes)
        if others:
            return out[(slice(None), *others)]
        return out

    def __array__(self, dtype=None):
        # more efficient
        out = np.zeros(self.shape, dtype=dtype)
        return self.as_array(out)

    def as_array(self, out=None):
        """TODO: docstring for `as_array`

        Args:
                out: [description], defaults to None
        """
        if out is None:
            out = np.zeros(self.shape)
        index = np.empty((self.nodes.max() + 1,), dtype=np.int64)
        index[self.nodes] = np.arange(len(self.nodes))
        count = np.zeros((len(self.nodes),), dtype=np.int64)
        for patch in tqdm(self.patches, desc="get full mean embedding"):
            nodes = patch.nodes
            out[index[nodes]] += patch.coordinates
            count[index[nodes]] += 1
        out /= count[:, None]
        return out

    def get_coordinates(self, nodes, out=None):
        """TODO: docstring for `get_coordinates`

        Args:

                nodes: [description]

                out: [description], default is None
        """
        nodes = np.asanyarray(nodes)
        if out is None:
            out = np.zeros((len(nodes), self.dim))

        count = np.zeros((len(nodes),), dtype=np.int)
        for patch in tqdm(self.patches, desc="get mean embedding"):
            index = [i for i, node in enumerate(nodes) if node in patch.index]
            count[index] += 1
            out[index] += patch.get_coordinates(nodes[index])
        out /= count[:, None]
        return out

    def __iadd__(self, other):
        for patch in self.patches:
            patch.coordinates += other
        return self

    def __isub__(self, other):
        for patch in self.patches:
            patch.coordinates -= other
        return self

    def __imul__(self, other):
        for patch in self.patches:
            patch.coordinates *= other
        return self

    def __itruediv__(self, other):
        for patch in self.patches:
            patch.coordinates /= other
        return self

    def __imatmul__(self, other):
        for patch in self.patches:
            patch.coordinates = patch.coordinates @ other
        return self

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        # TODO: review, this was changed from original code
        # new = self.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.patches = [copy.copy(patch) for patch in self.patches]
        return new

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.patches)})"
