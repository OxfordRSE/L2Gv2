"""TODO: module docstring for dgi/utils/loss.py."""

import torch_geometric as tg
import torch


# TODO: fix too-few-public-methods for dgi/utils/loss.py DGILoss
# pylint: disable=too-few-public-methods
class DGILoss(torch.nn.Module):
    """TODO: class docstring for DGILoss."""

    def __init__(self):
        super().__init__()
        self.loss_fun = torch.nn.BCEWithLogitsLoss()

    def forward(self, model, data: tg.data.Data):
        """TODO: method docstring for DGILoss.forward."""
        device = data.edge_index.device
        nb_nodes = data.num_nodes
        idx = torch.randperm(nb_nodes, device=device)

        shuf_fts = data.x[idx, :]

        lbl_1 = torch.ones(nb_nodes, device=device)
        lbl_2 = torch.zeros(nb_nodes, device=device)
        lbl = torch.cat((lbl_1, lbl_2), 0)

        logits = model(data.x, shuf_fts, data.edge_index, None, None, None)

        return self.loss_fun(logits, lbl)


# pylint: enable=too-few-public-methods
