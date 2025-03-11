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

    def forward(self, patch_emb):
        m = len(patch_emb)
        transformations = (
            [self.transformation[0]]
            + [
                item
                for i in range(1, len(self.transformation) - 1)
                for item in (self.transformation[i], self.transformation[i])
            ]
            + [self.transformation[-1]]
        )
        transformed_emb = [transformations[i](patch_emb[i]) for i in range(m)]
        return transformed_emb
