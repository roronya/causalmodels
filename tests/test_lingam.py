import numpy as np
import pandas as pd
import causalmodels as cm

a = np.random.laplace(size=100)
b = np.random.laplace(size=100)
c = np.random.laplace(size=100)
a = a + b + c
b = b + c
data = pd.DataFrame({"a":a, "b": b, "c": c})
model = cm.DirectLiNGAM(data.values, data.columns)
result = model.fit(regression="lasso")
print(result.order)
print(result.matrix)
result.plot(threshold=0.1)
