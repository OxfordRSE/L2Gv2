class VGAE:

    def __init__(self, model, loss_fun, lr: float=1e-2, num_epochs: int=100, verbose: bool=True, ):
        self.model = 

    def fit(data, model, loss_fun, num_epochs=100, verbose=True, lr=0.01, logger=lambda loss: None):
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

    def embed(self, patch_data, dim=100, hidden_dim=32, num_epochs=100, decoder=None, device='cpu', lr=0.01):
        patch_list = []
        models = []
        for i, patch in enumerate(patch_data):
            if patch.x is None:
                patch.x = speye(patch.num_nodes)
            print(f"Training patch {i} with {patch.edge_index.shape[1]} edges")   #added [i] to every patch
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
