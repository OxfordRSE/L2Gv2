""" TODO: module docstring for dgi/models/logreg.py. """
import torch
from torch import nn


class LogReg(nn.Module):
    """ TODO: class docstring for LogReg. """
    def __init__(self, ft_in, nb_classes):
        super().__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """ TODO: method docstring for LogReg.weights_init. """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        """ TODO: method docstring for LogReg.forward """
        ret = self.fc(seq)
        return ret
