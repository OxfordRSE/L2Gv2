import torch_geometric as tg
import torch
import torch.nn.functional as F
import local2global.utils as l2g


def speye(n, dtype=torch.float):
    """identity matrix of dimension n as sparse_coo_tensor."""
    return torch.sparse_coo_tensor(torch.tile(torch.arange(n, dtype=torch.long), (2, 1)),
                                   torch.ones(n, dtype=dtype),
                                   (n, n))

class DistanceDecoder(torch.nn.Module):
    def __init__(self):
        super(DistanceDecoder, self).__init__()
        self.dist = torch.nn.PairwiseDistance()

    def forward(self, z, edge_index, sigmoid=True):
        value = -self.dist(z[edge_index[0]], z[edge_index[1]])
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.cdist(z, z)
        return torch.sigmoid(adj) if sigmoid else adj


class GAEconv(torch.nn.Module):
    def __init__(self, dim, num_node_features, hidden_dim=32, cached=True, bias=True, add_self_loops=True, normalize=True):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(num_node_features, hidden_dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)
        self.conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.relu(self.conv1(data.x, edge_index))
        return self.conv2(x, edge_index)

def GAE_loss(model, data):
    return model.recon_loss(model.encode(data), data.edge_index)

class VGAEconv(torch.nn.Module):
    def __init__(self, dim, num_node_features, hidden_dim=32, cached=True, bias=True, add_self_loops=True, normalize=True):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(num_node_features, hidden_dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)
        self.mean_conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                        normalize=normalize)
        self.var_conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                       normalize=normalize)

    def forward(self, data: tg.data.Data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        mu = self.mean_conv2(x, edge_index)
        sigma = self.var_conv2(x, edge_index)
        return mu, sigma

    
def VGAE_loss(model, data):
    return model.recon_loss(model.encode(data), data.edge_index) + model.kl_loss() / data.num_nodes


def VGAE_model(dim, hidden_dim, num_features, dist=False):
    if dist:
        return tg.nn.VGAE(encoder=VGAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim),
                          decoder=DistanceDecoder())
    else:
        return tg.nn.VGAE(encoder=VGAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim))


def lr_grid_search(data, model, loss_fun, validation_loss_fun, lr_grid=(0.1, 0.01, 0.005, 0.001),
                   num_epochs=10, runs=1, verbose=True):
    val_loss = torch.zeros((len(lr_grid), runs))
    val_start = torch.zeros((len(lr_grid), runs))
    for i, lr in enumerate(lr_grid):
        for r in range(runs):
            model.reset_parameters()
            val_start[i, r] = validation_loss_fun(model, data)
            model = train(data, model, loss_fun, num_epochs=num_epochs, lr=lr, verbose=verbose)
            val_loss[i, r] = validation_loss_fun(model, data)
    model.reset_parameters()
    return lr_grid[torch.argmax(torch.mean(val_loss, 1))], val_loss, val_start


def train(data, model, loss_fun, num_epochs=100, verbose=True, lr=0.01, logger=lambda loss: None):
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
            print(f'epoch {e}: loss={loss.item()}')
        # schedule.step()
    return model


def VGAE_patch_embeddings(patch_data, dim=2, hidden_dim=32, num_epochs=100, decoder=None, device='cpu', lr=0.01):
    patch_list = []
    models = []
    for patch in patch_data:
        if patch.x is None:
            patch.x = speye(patch.num_nodes)
        print(f"training patch with {patch.edge_index.shape[1]} edges")
        model = tg.nn.VGAE(encoder=VGAEconv(dim, patch.x.shape[1], hidden_dim=hidden_dim), decoder=decoder).to(device)
        patch.to(device)

        def loss_fun(model, data):
            return model.recon_loss(model.encode(data), data.edge_index) + model.kl_loss() / data.num_nodes

        model = train(patch, model, loss_fun, num_epochs=num_epochs, lr=lr)
        with torch.no_grad():
            model.eval()
            coordinates = model.encode(patch).to('cpu').numpy()
            models.append(model)
            patch_list.append(l2g.Patch(patch.nodes.to('cpu').numpy(), coordinates))
    return patch_list, models


def GAE_patch_embeddings(patch_data, dim=2, hidden_dim=32, num_epochs=100, device='cpu', decoder=None, lr=0.01):
    patch_list = []
    models = []
    for patch in patch_data:
        if patch.x is None:
            patch.x = speye(patch.num_nodes)
        print(f"training patch with {patch.edge_index.shape[1]} edges")
        model = tg.nn.GAE(encoder=GAEconv(dim, patch.x.shape[1], hidden_dim=hidden_dim), decoder=decoder).to(device)
        patch.to(device)

        def loss_fun(model, data):
            return model.recon_loss(model.encode(data), data.edge_index)
        model.train()
        model = train(patch, model, loss_fun, num_epochs=num_epochs, lr=lr)
        model.eval()
        coordinates = model.encode(patch).to('cpu').data.numpy()
        patch.to('cpu')
        models.append(model)
        patch_list.append(l2g.Patch(patch.nodes.numpy(), coordinates))
    return patch_list, models


