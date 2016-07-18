import numpy as np
import pandas as pd
import causalmodels as cm

e1 = np.sin(np.arange(0, 30, 0.1))
e2 = np.random.laplace(size=300)
e3 = np.random.exponential(size=300)

x1 = e1
x2 = x1 + e2
x3 = x1 + x2 + e3

data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
model = cm.DirectLiNGAM(data.values, data.columns)
result = model.fit(regression="lasso")
print(result.order)
print(result.matrix)
result.plot(threshold=0.1, decimal=3)
