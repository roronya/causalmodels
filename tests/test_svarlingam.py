import numpy as np
import pandas as pd
import causalmodels as cm

a = np.random.laplace(size=100)
b = np.random.laplace(size=100) + a
c = np.random.laplace(size=100) + a + b
d = np.random.laplace(size=100) + a + b + c
b[1:] = b[1:] + a[:-1]
c[2:] = c[2:] + b[:-2] + a[:-2]
d[3:] = d[3:] + c[:-3] + b[:-3] + a[:-3]
data = pd.DataFrame({"a":a, "b": b, "c": c, "d": d})
model = cm.SVARDirectLiNGAM(data.values, data.columns)
result = model.fit(regression="lasso")
print(result.instantaneous_order)
print(result.matrixes)
result.plot(threshold=0.3)
result.plot(threshold=0.3, separate=True)
