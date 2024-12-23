# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""TODO: module docstring for dgi/models/dgi.py."""

from torch import nn
from ..layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    """TODO: class docstring for DGI."""

    def __init__(self, n_in, n_h, activation="prelu"):
        super().__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def reset_parameters(self):
        """TODO: method docstring for DGI.reset_parameters."""
        for m in self.children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2):
        """TODO: method docstring for DGI.forward."""
        h_1 = self.gcn(seq1, adj)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, data):
        """TODO: method docstring for DGI.embed."""
        h_1 = self.gcn(data.x, data.edge_index)

        return h_1.detach()
