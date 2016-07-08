import numpy as np
import pandas as pd
import causalmodels as cm

a = np.random.laplace(size=100)
b = np.random.laplace(size=100) + a
c = np.random.laplace(size=100) + a + b
data = pd.DataFrame({'a':a, 'b': b, 'c': c})
model = cm.DirectLiNGAM()
result = model.fit(data.values, data.columns, regression='lasso')
print(result.order)
print(result.matrix)
result.plot(threshold=1)
