"""
Train the model on the patch embeddings
"""

from torch import optim
import torch.nn.functional as F

from .model import OrthogonalModel, AffineModel
from ..utils import to_device


def patchgraph_mse_loss(transformed_emb):
    """
    Custom loss function that computes the squared norm of differences
    between transformed pairs in the dictionary.

    Args:
        transformed_dict: Dictionary with keys (i,j) and values (XW_i+b_i, YW_j+b_j)

    Returns:
        Total loss as the sum of squared differences
    """
    total_loss = 0.0

    for (_, _), (transformed_X, transformed_Y) in transformed_emb.items():
        # Calculate squared norm of the difference
        pair_loss = F.mse_loss(transformed_X, transformed_Y, reduction="sum")
        total_loss += pair_loss

    return total_loss


def train_alignment_model(
    patch_intersections,
    n_patches,
    device="cpu",
    num_epochs=100,
    learning_rate=0.05,
    model_type="affine",
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
    patch_emb = to_device(patch_intersections, device)
    dim = patch_emb[list(patch_emb.keys())[0]][0].shape[1]
    model = (
        AffineModel(dim, n_patches, device).to(device)
        if model_type == "affine"
        else OrthogonalModel(dim, n_patches, device).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_hist = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        transformed_patch_emb = model(patch_emb)
        loss = patchgraph_mse_loss(transformed_patch_emb)
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_hist.append(loss.item())
        if verbose:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss_hist
