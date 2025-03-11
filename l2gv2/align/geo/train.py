"""
Train the model on the patch embeddings
"""

import torch
from torch import optim
from .model import OrthogonalModel


def loss_function(transformed_emb):
    m = len(transformed_emb)
    diff = [transformed_emb[i] - transformed_emb[i + 1] for i in range(0, m - 1, 2)]
    loss = sum(torch.norm(d) ** 2 for d in diff)
    return loss


def train_alignment_model(
    patch_emb,
    n_patches,
    device="cpu",
    num_epochs=100,
    learning_rate=0.05,
    verbose=True,
):
    """
    Train the model on the patch embeddings
    Args:
        patch_emb: list of torch.Tensor
            patch embeddings
        n_patches: int
            number of patches
        device: str
            device to run the model on
        num_epochs: int
            number of epochs to train the model
        learning_rate: float
            learning rate for the optimizer
    Returns:
        model: Model
        loss_hist: list
    """
    patch_emb = [p.to(device).to(torch.float32) for p in patch_emb]
    dim = patch_emb[0].shape[1]
    model = OrthogonalModel(dim, n_patches, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_hist = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        transformed_patch_emb = model(patch_emb)
        loss = loss_function(transformed_patch_emb)
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_hist.append(loss.item())
        if verbose:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss_hist
