"""TODO: module docstring for dgi/layers/readout.py."""

import torch
from torch import nn

# TODO: fix too-few-public-methods for embedding/dgi/layers/readout.py AvgReadout
# pylint: disable=too-few-public-methods
# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    """TODO: class docstring for AvgReadout."""


    def forward(self, seq, msk):
        """TODO: method docstring for AvgReadout.forward."""
        if msk is None:
            return torch.mean(seq, 0)

        msk = torch.unsqueeze(msk, -1)
        return torch.sum(seq * msk, 0) / torch.sum(msk)
# pylint: enable=too-few-public-methods
