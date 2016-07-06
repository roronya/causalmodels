import numpy as np
import pandas as pd
import causalmodels as cm

a = np.random.laplace(size=300)
b = np.random.laplace(size=300) + a
c = np.random.laplace(size=300) + a + b
data = pd.DataFrame({'a':a, 'b': b, 'c': c})
model = cm.DirectLiNGAM()
result = model.fit(data.values, data.columns, regression='lasso', threshold=1)
print(result.order)
print(result.sorted_matrix)
result.plot()
