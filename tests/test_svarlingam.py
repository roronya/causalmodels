import numpy as np
import pandas as pd
import causalmodels as cm

a = np.random.laplace(size=300)
b = np.random.laplace(size=300) + a
c = np.random.laplace(size=300) + a + b
b[1:] = b[1:] + a[:-1]
c[2:] = c[2:] + b[:-2] + a[:-2]
data = pd.DataFrame({'a':a, 'b': b, 'c': c})
model = cm.SVARDirectLiNGAM()
result = model.fit(data.values, data.columns, 'lasso')
print(result.order)
print(result.sorted_matrix)
result.plot()
