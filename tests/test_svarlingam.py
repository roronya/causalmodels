import numpy as np
import pandas as pd
import causalmodels as cm

e0 = np.sin(np.arange(0, 30, 0.1))
e1 = np.random.laplace(size=300)
e2 = np.random.exponential(size=300)

x0 = e0
x1 = 0.5 * x0 + e1
x2 = 0.5 * x0 + 0.5 * x1 + e2
x_t = np.array([x0, x1, x2])[:, 1:]
x_t_1 = np.array([x0, x1, x2])[:, :-1]
B = np.array([[0, 0, 0.5],
              [0, 0.5, 0],
              [0.5, 0, 0]])
X = x_t + np.dot(B, x_t_1)

data = pd.DataFrame({'x0': X[0], 'x1': X[1], 'x2': X[2]})
model = cm.SVARDirectLiNGAM(data.values, data.columns)
result = model.fit(regression="lasso", maxlags=1)
print(result.matrixes)
result.plot(threshold=0.2)
