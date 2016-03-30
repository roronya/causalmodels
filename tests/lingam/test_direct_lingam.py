import numpy as np
import causalmodels.api as cm

a = np.random.laplace(size=100)
b = np.random.laplace(size=100) + a
c = np.random.laplace(size=100) + a + b
data = np.array([c, b, a])

#a = np.array([1 for i in range(100)])
#b = np.array([2 for i in range(100)])
#c = np.array([3 for i in range(100)])
#data = np.array([c, b, a])

direct_lingam = cm.DirectLiNGAM()
results = direct_lingam.fit(data)
print(results.get_causal_order())
print(results.get_causal_inference_matrix())
