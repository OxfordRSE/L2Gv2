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

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""TODO: module docstring for embedding/train.py."""

import torch
import torch_geometric as tg
from typing import Callable

from ..utils import EarlyStopping


def lr_grid_search(
    data,
    model,
    loss_fun,
    validation_loss_fun,
    lr_grid=(0.1, 0.01, 0.005, 0.001),
    num_epochs=10,
    runs=1,
    verbose=True,
):
    """
    grid search over learning rate values

    Args:
        data: input data
        model: model to train
        loss_fun: training loss takes model and data as input
        validation_loss_fun: function to compute validation loss input: (model, data)
        lr_grid: learning rate values to try
        num_epochs: number of epochs for training
        runs: number of training runs to average over for selecting best learning rate
        verbose: if ``True``, output training progress

    Returns:
        best learning rate, validation loss for all runs
    """
    val_loss = torch.zeros((len(lr_grid), runs))
    for i, lr in enumerate(lr_grid):
        for r in range(runs):
            model.reset_parameters()
            model = train(
                data, model, loss_fun, num_epochs=num_epochs, lr=lr, verbose=verbose
            )
            val_loss[i, r] = validation_loss_fun(model, data)
    model.reset_parameters()
    return lr_grid[torch.argmax(torch.mean(val_loss, 1))], val_loss


def train(
    data: tg.data.Data,
    model: torch.nn.Module,
    loss_fun: Callable,
    num_epochs: int = 10000,
    patience: int = 20,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    verbose: bool = True,
    logger: Callable = lambda loss: None,
):
    """
    Train an embedding model

    Args:
        data: input data
        model: model to train
        loss_fun: loss function
        num_epochs: number of epochs to train
        patience: patience for early stopping
        lr: learning rate
        weight_decay: weight decay
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    with EarlyStopping(patience) as stop:
        for e in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fun(model, data)
            f_loss = float(loss)
            logger(f_loss)
            if verbose:
                print(f"epoch {e}: loss={f_loss}")
            if stop(f_loss, model):
                if verbose:
                    print(f"Early stopping at epoch {e}")
                break
            loss.backward()
            optimizer.step()
    return model
