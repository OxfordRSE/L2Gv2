"""TODO: module docstring for dgi/layers/__init__.py."""

from .gcn import GCN
from .readout import AvgReadout
from .discriminator import Discriminator

__all__ = ["GCN", "AvgReadout", "Discriminator"]
