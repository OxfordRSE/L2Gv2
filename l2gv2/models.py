"""This module contains the functions to compute the
embeddings of a list of patches using VGAE and Node2Vec."""

from typing import Tuple
import numpy.typing as npt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import Node2Vec

from l2gv2.patch.patch import Patch
from l2gv2.patch.utils import WeightedAlignmentProblem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def speye(n: int, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Returns the identity matrix of dimension n as torch.sparse_coo_tensor.

    Args:
        n: dimension of the identity matrix.

        dtype: data type of the identity matrix, default is torch.float.

    Returns:
        identity matrix of dimension n as torch.sparse_coo_tensor.

    """

    return torch.sparse_coo_tensor(
        torch.tile(torch.arange(n, dtype=torch.long), (2, 1)),
        torch.ones(n, dtype=dtype),
        (n, n),
    )


# TODO: fix too-few-public-methods
# pylint: disable=too-few-public-methods
class VGAEconv(torch.nn.Module):
    """TODO: docstring for `VGAEconv`"""

    def __init__(
        self,
        dim: int,
        num_node_features: int,
        hidden_dim: int = 32,
        cached: bool = True,
        bias=True,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(
            num_node_features,
            hidden_dim,
            cached=cached,
            bias=bias,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )
        self.mean_conv2 = tg.nn.GCNConv(
            hidden_dim,
            dim,
            cached=cached,
            bias=bias,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )
        self.var_conv2 = tg.nn.GCNConv(
            hidden_dim,
            dim,
            cached=cached,
            bias=bias,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )

    def forward(self, data: tg.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            data: object with edge_index and x attributes.

        Returns:
            mu: mean of the latent space.

            sigma: variance of the latent space.
        """
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        mu = self.mean_conv2(x, edge_index)
        sigma = self.var_conv2(x, edge_index)
        return mu, sigma


# pylint: enable=too-few-public-methods


def train(
    data: tg.data.Data,
    model: torch.nn.Module,
    loss_fun,
    num_epochs: int = 100,
    verbose: bool = True,
    lr: float = 0.01,
    logger=lambda loss: None,
) -> torch.nn.Module:
    """Train a model on a dataset.

    Args:
        data: object to train the model on.

        model: the model to train.

        loss_fun: function that takes the model and the data and returns a scalar loss.

        num_epochs: number of epochs to train the model, default is 100.

        verbose: if True, print the loss at each epoch, default is True.

        lr: learning rate, default is 0.01.

        logger: function that takes the loss and logs it.

    Returns:
        trained model
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fun(model, data)
        loss.backward()
        optimizer.step()
        logger(float(loss))
        if verbose:
            print(f"epoch {e}: loss={loss.item()}")
        # schedule.step()
    return model


def vgae_patch_embeddings(
    patch_data: list,
    dim: int = 100,
    hidden_dim: int = 32,
    num_epochs: int = 100,
    decoder=None,
    lr: float = 0.01,
) -> Tuple[list[Patch], list[torch.nn.Module]]:
    """TODO: docstring for `vgae_patch_embeddings`

    Args:
        patch_data: list of torch_geometric.data.Data objects.

        dim: dimension of the latent space, default is 100.

        hidden_dim: hidden dimension of the encoder, default is 32.

        num_epochs: number of epochs to train the model, default is 100.

        decoder : decoder function, default is None.

        lr : learning rate, default is 0.01.

    Returns:
        list of Patch objects.

        list of trained models.
    """

    patch_list = []
    models = []
    for i, patch in enumerate(patch_data):
        if patch.x is None:
            patch.x = speye(patch.num_nodes)
        print(
            f"training patch {i} with {patch.edge_index.shape[1]} edges"
        )  # added [i] to every patch
        model = tg.nn.VGAE(
            encoder=VGAEconv(dim, patch.x.shape[1], hidden_dim=hidden_dim),
            decoder=decoder,
        ).to(device)
        patch.to(device)

        def loss_fun(model, data):
            return (
                model.recon_loss(model.encode(data), data.edge_index)
                + model.kl_loss() / data.num_nodes
            )

        model = train(patch, model, loss_fun, num_epochs=num_epochs, lr=lr)
        with torch.no_grad():
            model.eval()
            coordinates = model.encode(patch).to("cpu").numpy()
            models.append(model)
            patch_list.append(Patch(patch.nodes.to("cpu").numpy(), coordinates))
    return patch_list, models


def node2vec_(
    data: tg.data.Data,
    emb_dim: int,
    w_length: int = 20,
    c_size: int = 10,
    w_per_node: int = 10,
    n_negative_samples: int = 1,
    p: int = 1,
    q: int = 1,
    num_epochs: int = 100,
) -> Patch:
    """TODO: docstring for `node2vec_`

    Args:

        data: [description]

        emb_dim: [description]

        w_length: The walk length, default is 20.

        c_size: Actual context size which is considered for
            positive samples, default is 10.

        w_per_node: The number of walks per node, default is 10.

        n_negative_samples: The number of negative samples to
            use for each positive sample., default is 1.

        p: likelihood of immediately revisiting a node in the
            walk, default is 1.

        q: Control parameter to interpolate between
            breadth-first strategy and depth-first strategy, default is 1.

        num_epoch: number of epochs to train the model, default is 100.

    Returns:

        [description]
    """

    node2vec_model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=emb_dim,
        walk_length=w_length,
        context_size=c_size,
        walks_per_node=w_per_node,
        num_negative_samples=n_negative_samples,
        p=p,
        q=q,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(node2vec_model.parameters(), lr=0.01)

    # Data loader for random walks
    loader = node2vec_model.loader(batch_size=128, shuffle=True, num_workers=0)

    # Training loop

    node2vec_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader)}")

    # Get embeddings after training
    node_embeddings = node2vec_model.embedding.weight.data.cpu()
    # models.append(model)

    return Patch(data.nodes.to("cpu").numpy(), node_embeddings)


def node2vec_patch_embeddings(
    patch_data: list,
    emb_dim,
    w_length=20,
    c_size=10,
    w_per_node=10,
    n_negative_samples=1,
    p=1,
    q=1,
) -> list[Patch]:
    """TODO: docstring for `node2vec_patch_embeddings`

    Args:

        patch_data: torch_geometric.data.Data objects.

        emb_dim: [description]

        w_length: The walk length, default is 20.

        c_size: Actual context size which is considered for
            positive samples, default is 10.

        w_per_node: The number of walks per node, default is 10.

        n_negative_samples: The number of negative samples to
            use for each positive sample., default is 1.

        p: likelihood of immediately revisiting a node in the
            walk, default is 1.

        q: Control parameter to interpolate between
            breadth-first strategy and depth-first strategy, default is 1.

    Returns:

        list of Patch objects.
    """

    patch_list = []

    for i, data in enumerate(patch_data):
        print(f"training patch {i+1} with {data.edge_index.shape[1]} edges")

        patch_list.append(
            node2vec_(
                data,
                emb_dim,
                w_length=w_length,
                c_size=c_size,
                w_per_node=w_per_node,
                n_negative_samples=n_negative_samples,
                p=p,
                q=q,
            )
        )

    return patch_list


def chunk_embedding(
    chunk_size: int, patches: list[Patch], dim=2
) -> Tuple[npt.ArrayLike, WeightedAlignmentProblem]:
    """TODO: docstring for `chunk_embedding`

    Note: this only works for Autonomous System dataset.

    Args:

        chunk_size: The size of the chunks.

        patches: list of Patch objects.

        dim: The dimension of the embeddings, default is 2.

    Returns:

        embeddings of the nodes.

        WeightedAlignmentProblem object.

    """

    t_all = [from_networkx(g) for g in patches]
    list_set_nodes = [set(patches[i].nodes()) for i in range(len(patches))]

    nodes_in_intersection = set.intersection(*[set(gr) for gr in list_set_nodes])

    nodes_in_union = set()
    for s in list_set_nodes:
        nodes_in_union = nodes_in_union.union(s)

    missing_nodes = set(i for i in range(max(nodes_in_union)))
    for i in range(max(nodes_in_union)):
        if i in nodes_in_union:
            missing_nodes.remove(i)

    for i, p in tqdm(enumerate(t_all)):
        p.nodes = torch.Tensor(list(patches[i].nodes)).int()
        p.num_nodes = p.nodes.size(0)
    sub_patches = [t_all[i : i + chunk_size] for i in range(0, len(t_all), chunk_size)]

    nodes_in_intersection = []
    list_nodes_tot = []
    for g in tqdm(sub_patches):
        nodes_in_each_p = [set(p.nodes.tolist()) for p in g]
        nodes_in_intersection.append(set.intersection(*nodes_in_each_p))
        nodes_tot = []
        for p in nodes_in_each_p:
            nodes_tot += p
        list_nodes_tot.append(set(nodes_tot))

    emb = []
    for i, patch in tqdm(enumerate(sub_patches)):
        p_emb = vgae_patch_embeddings(
            patch,
            dim=dim,
            hidden_dim=32,
            num_epochs=100,
            decoder=None,
            lr=0.01,
        )

        prob = WeightedAlignmentProblem(
            p_emb[0]
        )  # embedding of the full graph using embeddings of each patch

        emb.append(prob.get_aligned_embedding())
    ppatch2 = [
        Patch(list(list_nodes_tot[i]), emb[i][list(list_nodes_tot[i])])
        for i in range(len(emb))
    ]

    prob = WeightedAlignmentProblem(ppatch2)

    emb = prob.get_aligned_embedding()

    nodes_emb = emb[list(nodes_in_union)]

    return nodes_emb, prob
