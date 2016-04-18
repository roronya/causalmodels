import numpy as np
import causalmodels as cm

a = np.random.laplace(size=100)
b = np.random.laplace(size=100) + a
c = np.random.laplace(size=100) + a + b
data = np.array([a,b,c])
labels = ['a','b','c']

model = cm.SparseDirectLiNGAM()
result = model.fit(data, labels)
result.draw()
