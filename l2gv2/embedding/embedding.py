"""
Code for embedding individual patches using VGAE or GAE.
"""


class EmbeddingModel:
    """
    Base class for embedding models.
    """

    def __init__(self, model, dim):
        self.model = model
        self.dim = dim

    def fit(self, data):
        pass

    def transform(self, data):
        pass

    def fit_transform(self, data):
        pass
