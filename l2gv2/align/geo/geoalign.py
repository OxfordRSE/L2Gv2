"""
Alignment using pytorch geometric.
"""

from l2gv2.align.registry import register_aligner
from l2gv2.align.alignment import AlignmentProblem
from l2gv2.align.utils import preprocess_graphs, intersections_nodes
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
        num_epoches: int = 100,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__(
            patches, patch_edges, min_overlap, copy_data, self_loops, verbose
        )
        self.num_epochs = num_epoches
        self.learning_rate = learning_rate
        self.device = device

    def align_patches(self, scale=False):  # pylint: disable=unused-argument
        """
        Align the patches.
        """
        n_patches = len(self.patches)
        nodes = intersections_nodes(self.patches)
        emb_patches = preprocess_graphs(self.patches, nodes)
        res, _ = train_alignment_model(
            emb_patches,
            n_patches,
            device=self.device,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )

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
