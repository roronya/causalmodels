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
    >>> import causalmodels as cm
    >>> a = np.random.laplace(size=500)
    >>> b = np.random.laplace(size=500) + a
    >>> c = np.random.laplace(size=500) + a + b
    >>> data = np.array([c, b, a])
    >>>
    >>> model = cm.DirectLiNGAM()
    >>> results = model.fit(data)
    >>> results.order
    [2, 1, 0]
    >>> result.draw()
