class ModelInterface:
    """
    Interface of model.
    """
    def fit(self):
        """
        fit to model.
        """
        raise NotImplementedError()


class ResultInterface:
    """
    Interface of results.
    model must be return Results obejct.
    """
    def plot(self):
        raise NotImplementedError()
