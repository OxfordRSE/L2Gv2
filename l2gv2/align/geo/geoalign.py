"""
Alignment using pytorch geometric.
"""

from l2gv2.align.registry import register_aligner
from l2gv2.align.alignment import AlignmentProblem
from l2gv2.align.utils import get_intersections
from l2gv2.align.geo.train import train_alignment_model
from l2gv2.patch import Patch


@register_aligner("geo")
class GeoAlignmentProblem(AlignmentProblem):
    """
    Alignment problem using pytorch geometric.
    """

    def __init__(
        self,
        patches: list[Patch],
        patch_edges=None,
        min_overlap=None,
        copy_data=True,
        self_loops=False,
        verbose=False,
        num_epochs: int = 1000,
        learning_rate: float = 0.001,
        model_type: str = "affine",
        device: str = "cpu",
    ):
        super().__init__(
            patches, patch_edges, min_overlap, copy_data, self_loops, verbose
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model_type = model_type
        self.loss_hist = []

    def align_patches(self, scale=False):  # pylint: disable=unused-argument
        """
        Align the patches.
        """
        n_patches = len(self.patches)
        _, embeddings = get_intersections(self.patches)

        res, loss_hist = train_alignment_model(
            embeddings,
            n_patches,
            device=self.device,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            model_type=self.model_type,
            verbose=self.verbose,
        )

        self.loss_hist = loss_hist

        self.rotations = [
            res.transformation[i].weight.to("cpu").detach().numpy()
            for i in range(n_patches)
        ]
        self.shifts = [
            res.transformation[i].bias.to("cpu").detach().numpy()
            for i in range(n_patches)
        ]
        for i, patch in enumerate(self.patches):
            self.patches[i].coordinates = patch.coordinates @ self.rotations[i].T
            self.patches[i].coordinates += self.shifts[i]
        return self
