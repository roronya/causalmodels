# causalmodels
causalmodels in Python.

## instalation

    $ pip install causalmodels

## usage

    >>> import numpy as np
    >>> import causalmodels.api as cm
    >>> a = np.random.laplace(size=500)
    >>> b = np.random.laplace(size=500) + a
    >>> c = np.random.laplace(size=500) + a + b
    >>> data = [c, b, a]
    >>>
    >>> model = cm.DirectLiNGAM()
    >>> results = model.fit(data)
    >>> results.get_causal_order()
    [2, 1, 0]
