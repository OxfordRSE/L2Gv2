"""Tests for l2gv2.patch.sparsify"""

import torch

from l2gv2.patch.sparsify import (
    _gumbel_topk,
    _sample_edges,
    conductance_weighted_graph,
    resistance_weighted_graph,
)

# _gumbel_topk tests


def test_gumbel_topk_k_equals_len():
    weights = torch.tensor([0.2, 0.5, 0.3])
    k = len(weights)
    result = _gumbel_topk(weights, k)
    expected = torch.arange(len(weights))
    assert torch.equal(torch.sort(result).values, expected)


def test_gumbel_topk_output_size_and_type():
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    k = 2
    result = _gumbel_topk(weights, k)
    assert result.shape == (k,)
    assert result.dtype == torch.long
    assert result.max() < len(weights)
    assert result.min() >= 0


def test_gumbel_topk_with_log_weights_true():
    weights = torch.tensor([0.1, 0.9, 0.5, 0.7])
    k = 2
    result = _gumbel_topk(weights, k, log_weights=True)
    assert result.shape == (k,)
    assert result.dtype == torch.long


def test_gumbel_topk_invalid_k():
    weights = torch.tensor([0.1, 0.2])
    k = 5
    result = _gumbel_topk(weights, k)
    assert torch.equal(result, torch.arange(len(weights)))


def test_sample_edges(toy_tgraph):
    # TODO: check edge cases
    assert (
        _sample_edges(toy_tgraph, n_desired_edges=2)
        == torch.Tensor([True, True, True, False])
    ).all()


# TODO: Currently these are only snapshot tests and do not look at edge cases
def test_conductance_weighted_graph(toy_tgraph_weighted):
    cw = conductance_weighted_graph(toy_tgraph_weighted)
    assert torch.allclose(
        torch.Tensor(cw.weights), torch.Tensor([1.0, 1.33333333, 1.0, 0.33333333])
    )


def test_resistance_weighted_graph(toy_tgraph_weighted):
    rw = resistance_weighted_graph(toy_tgraph_weighted)
    assert torch.allclose(
        torch.Tensor(rw.weights),
        torch.DoubleTensor([0.5833333, 0.6666667, 0.5833333, 0.1666667]),
    )
