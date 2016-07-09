import numpy as np
import pandas as pd
import causalmodels as cm

a = np.random.laplace(size=100)
b = np.random.laplace(size=100) + a
c = np.random.laplace(size=100) + a + b
b[1:] = b[1:] + a[:-1]
c[2:] = c[2:] + b[:-2] + a[:-2]
data = pd.DataFrame({"a":a, "b": b, "c": c})
model = cm.SVARDirectLiNGAM(data.values, data.columns)
result = model.fit(regression="lasso")
print(result.instantaneou_order)
print(result.matrixes)
result.plot(threshold=0.3)
