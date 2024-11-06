""" Module for embedding patches using the VGAE model """
import torch
import torch_geometric as tg
from l2gv2.models import speye, VGAEconv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGAE:
    """ TODO: docstring for `VGAE` """
    def __init__(
        self,
        model,
        lr: float = 1e-2,
        num_epochs: int = 100,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.lr = lr

    def fit(
        self,
        data,
        logger=lambda loss: None,
    ):
        """ Train the model on the given data 
        
        Args:

            data (torch_geometric.data.Data):

            logger (): 

        Returns:
            
            torch.nn.Module: 
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        for e in range(self.num_epochs):
            optimizer.zero_grad()
            loss = self.loss_fun(data)
            loss.backward()
            optimizer.step()
            logger(float(loss))
            if self.verbose:
                print(f"epoch {e}: loss={loss.item()}")
            # schedule.step()
        return self.model

    def embed(
        self,
        patch_data: list,
        dim: int=100,
        hidden_dim: int=32,
        decoder=None
    ):

        """ TODO: docstring for `embed` 
        Args:

            patch_data (list): list of torch_geometric.data.Data

            dim (int, optional): defaults to 100 

            hidden_dim (int, optional): defaults to 32

            decoder (): defaults to None


        Returns:
            
            list of l2gv2.Patch: list of patches with embedded coordinates

            list of torch.nn.Module: list of trained models
        """
        patch_list = []
        models = []
        for i, patch in enumerate(patch_data):
            if patch.x is None:
                patch.x = speye(patch.num_nodes)
            print(
                f"Training patch {i} with {patch.edge_index.shape[1]} edges"
            )  # added [i] to every patch
            model = tg.nn.VGAE(
                encoder=VGAEconv(dim, patch.x.shape[1], hidden_dim=hidden_dim),
                decoder=decoder,
            ).to(device)
            patch.to(device)

            # TODO: add actual code, these are just placeholder statements to avoid linting errors
            models.append(model)
            patch_list.append(patch)
            return patch_list, models


    def loss_fun(self, data):
        """ TODO: docstring for `loss_fun`
        
        Args:

            data (torch_geometric.data.Data)

        Returns:
                
            torch.Tensor: 
        """
        return (
            self.model.recon_loss(self.model.encode(data), data.edge_index)
            + self.model.kl_loss() / data.num_nodes
        )


        # TODO: clarify what this code does and where it should be placed
        #
        # model = train(patch, model, loss_fun, num_epochs=num_epochs, lr=lr)
        # with torch.no_grad():
        #     model.eval()
        #     coordinates = model.encode(patch).to("cpu").numpy()
        #     models.append(model)
        #     patch_list.append(l2g.Patch(patch.nodes.to("cpu").numpy(), coordinates))

        # return patch_list, models
