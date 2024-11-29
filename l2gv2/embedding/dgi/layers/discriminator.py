"""TODO: module docstring for dgi/layers/discriminator.py."""

import torch
from torch import nn


class Discriminator(nn.Module):
    """TODO: class docstring for Discriminator."""

    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """TODO: method docstring for Discriminator.reset_parameters."""
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """TODO: method docstring for Discriminator.forward."""
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits
