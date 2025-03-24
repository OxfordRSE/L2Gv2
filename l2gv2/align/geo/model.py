"""
Model for aligning patch embeddings
"""

from torch import nn
import geotorch


# pylint: disable=too-few-public-methods
class OrthogonalModel(nn.Module):
    """
    Model for aligning patch embeddings
    """

    def __init__(self, dim, n_patches, device):
        """
        Initialize the model
        Args:
            dim: int
            n_patches: int
            device: str
        """
        super().__init__()
        self.device = device
        self.transformation = nn.ParameterList(
            [nn.Linear(dim, dim).to(device) for _ in range(n_patches)]
        )
        for i in range(n_patches):
            geotorch.orthogonal(self.transformation[i], "weight")

    def forward(self, patch_intersection):
        """
        Forward pass
        Args:
            patch_intersection: list of tuples
        Returns:
            list of transformed embeddings
        """
        outputs = {}
        for (i, j), (X, Y) in patch_intersection.items():
            Xt = self.transformation[i](X)
            Yt = self.transformation[j](Y)
            outputs[(i, j)] = (Xt, Yt)
        return outputs


# pylint: disable=too-few-public-methods
class AffineModel(nn.Module):
    """
    Model for aligning patch embeddings
    """

    def __init__(self, dim, n_patches, device):
        """
        Initialize the model
        Args:
            dim: int
            n_patches: int
            device: str
        """
        super().__init__()
        self.device = device
        self.transformation = nn.ParameterList(
            [nn.Linear(dim, dim, bias=True).to(device) for _ in range(n_patches)]
        )

    def forward(self, patch_intersection):
        """
        Forward pass
        """
        outputs = {}
        for (i, j), (X, Y) in patch_intersection.items():
            Xt = self.transformation[i](X)
            Yt = self.transformation[j](Y)
            outputs[(i, j)] = (Xt, Yt)
        return outputs
