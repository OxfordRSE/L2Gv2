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
import torch
import itertools

rg = np.random.default_rng()
eps = np.finfo(float).eps


def seed(new_seed):
    """
    Change seed of random number generator.

    Args:
        new_seed: New seed value

    Returns:
        New random number generator instance

    """
    return np.random.default_rng(new_seed)


def preprocess_graphs(list_of_patches, nodes_dict):
    """Preprocess the graphs to get the embedding of the patches."""
    emb_list = []
    for i in range(len(list_of_patches) - 1):
        emb_list.append(
            [
                torch.tensor(
                    list_of_patches[i].get_coordinates(list(nodes_dict[i, i + 1]))
                ),
                torch.tensor(
                    list_of_patches[i + 1].get_coordinates(list(nodes_dict[i, i + 1]))
                ),
            ]
        )
    emb_list = list(itertools.chain.from_iterable(emb_list))
    return emb_list


def intersections_nodes(patches):
    """Calculate the intersection of nodes between patches."""
    intersections = {}
    for i, _ in enumerate(patches):
        for j in range(i + 1, len(patches)):
            intersections[(i, j)] = list(
                set(patches[i].nodes.tolist()).intersection(
                    set(patches[j].nodes.tolist())
                )
            )
    return intersections
