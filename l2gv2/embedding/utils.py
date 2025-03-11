"""
Code for embedding individual patches using VGAE or GAE.
"""

import torch
import numpy as np
import torch_geometric as tg
from l2gv2.patch import Patch
from l2gv2.embedding.gae import GAE, VGAE
from l2gv2.embedding.train import train
from l2gv2.utils import speye
from l2gv2.align.alignment import AlignmentProblem


def patch_embeddings(
    patch_data: list[tg.data.Data],
    model: str = "VGAE",
    dim: int = 64,
    hidden_dim: int = 128,
    num_epochs: int = 100,
    device: str = "cpu",
    lr: float = 0.01,
):
    """
    Embed patches using VGAE or GAE.

    Args:
        patch_data: List of patches to embed.
        model: Model to use for embedding.
        dim: Dimension of the embedding.
        hidden_dim: Hidden dimension of the embedding.
        num_epochs: Number of epochs to train the model.
    """
    patch_list = []
    models = []
    for patch in patch_data:
        if patch.x is None:
            patch.x = speye(patch.num_nodes)
        print(
            f"training patch with {patch.edge_index.shape[1]} edges"
        )  # added [i] to every patch
        if model == "VGAE":
            embedding_model = VGAE(dim, hidden_dim, patch.x.shape[1]).to(device)
        elif model == "GAE":
            embedding_model = GAE(dim, hidden_dim, patch.x.shape[1]).to(device)
        else:
            raise ValueError(f"Invalid model: {model}")
        patch.to(device)

        def loss_fun(model, data):
            return (
                model.recon_loss(model.encode(data), data.edge_index)
                + model.kl_loss() / data.num_nodes
            )

        embedding_model = train(
            patch,
            embedding_model,
            loss_fun,
            num_epochs=num_epochs,
            verbose=False,
            lr=lr,
        )
        with torch.no_grad():
            embedding_model.eval()
            coordinates = embedding_model.encode(patch).to("cpu").numpy()
            models.append(embedding_model)
            patch_list.append(Patch(patch.nodes.to("cpu").numpy(), coordinates))
    return patch_list, models


def get_embedding(patches: list[Patch], result: AlignmentProblem):
    """
    Get the embedding of the patches after alignment.

    Args:
        patches: List of patches to embed.
        result: Result of the alignment.
    """
    n = len(patches)
    rot = [result.transformation[i].weight.to("cpu").detach().numpy() for i in range(n)]
    shift = [result.transformation[i].bias.to("cpu").detach().numpy() for i in range(n)]

    emb_problem = AlignmentProblem(patches)
    embedding = np.empty((emb_problem.n_nodes, emb_problem.dim))
    for node, patch_list in enumerate(emb_problem.patch_index):
        embedding[node] = np.mean(
            [
                emb_problem.patches[p].get_coordinate(node) @ rot[i] + shift[i]
                for i, p in enumerate(patch_list)
            ],
            axis=0,
        )
    return embedding
