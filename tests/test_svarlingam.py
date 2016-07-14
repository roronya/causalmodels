import numpy as np
import pandas as pd
import causalmodels as cm

a = 0.5
b = 1
v1org = np.random.laplace(size=102)
v2org = np.random.laplace(size=102)
v3org = np.random.laplace(size=102)
v1 = v1org[2:] + a * v1org[:-2] + b * v1org[1:-1]
v2 = v2org[2:] + a * v2org[:-2] + b * v2org[1:-1] + a * v1org[:-2] + b * v1org[2:]
v3 = v3org[2:] + a * v3org[:-2] + b * v3org[1:-1]
data = pd.DataFrame({"v1": v1, "v2": v2, "v3": v3})
data = (data - data.mean()) / data.std(ddof=False)
model = cm.SVARDirectLiNGAM(data.values, data.columns)
result = model.fit(regression="lasso", maxlags=2)
print(result.instantaneous_order)
print(result.matrixes)
result.plot(threshold=0.2)
