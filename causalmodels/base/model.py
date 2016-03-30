class Model:
    """
    Interface of model.
    """

    def __init__(self):
        pass

    def fit(self):
        """
        fit to model.
        """
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class Results:
    """
    Interface of results.
    model must be return Results obejct.
    """

    def __init__(self, causal_order, causal_inference_matrix):
        self.causal_order = causal_order
        self.causal_inference_matrix = causal_inference_matrix

    def get_causal_order(self):
        return self.causal_order

    def get_causal_inference_matrix(self):
        return self.causal_inference_matrix
