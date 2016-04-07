import numpy as np

class ModelInterface:
    """
    Interface of model.
    """
    def fit(self):
        """
        fit to model.
        """
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class ResultInterface:
    """
    Interface of results.
    model must be return Results obejct.
    """
    def draw():
        raise NotImplementedError()
