causalmodels
============

causalmodels in Python.

instalation
-----------

::

    $ pip install causalmodels

usage
-----

::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import causalmodels as cm
    >>> a = np.random.laplace(size=500)
    >>> b = np.random.laplace(size=500) + a
    >>> c = np.random.laplace(size=500) + a + b
    >>> data = pd.DataFrame({'a': a, 'b': b, 'c': c})
    >>> model = cm.DirectLiNGAM(data.values, data.columns)
    >>> results = model.fit()
    >>> results.order
    [2, 1, 0]
    >>> result.plot()
