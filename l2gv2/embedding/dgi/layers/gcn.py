""" TODO: module docstring for dgi/layers/gcn.py. """
from torch import nn
import torch_geometric.nn as tg_nn


class GCN(nn.Module):
    """ TODO: class docstring for GCN. """
    def __init__(self, in_ft, out_ft, act, bias=True):
        super().__init__()
        self.conv = tg_nn.GCNConv(in_channels=in_ft, out_channels=out_ft, bias=bias)
        self.act = nn.PReLU() if act == "prelu" else act
        self.reset_parameters()

    def reset_parameters(self):
        """ TODO: method docstring for GCN.reset_parameters. """
        self.conv.reset_parameters()
        if hasattr(self.act, "reset_parameters"):
            self.act.reset_parameters()
        elif isinstance(self.act, nn.PReLU):
            self.act.weight.data.fill_(0.25)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        """ TODO: method docstring for GCN.forward. """
        out = self.conv(seq, adj)

        return self.act(out)
