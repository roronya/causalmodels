import unittest
import numpy as np
import pandas as pd
import causalmodels as cm

class TestSVARLiNGAMMethods(unittest.TestCase):
    def setUp(self):
        e0 = np.sin(np.arange(0, 30, 0.1))
        e1 = np.random.laplace(size=300)
        e2 = np.random.exponential(size=300)
        e = np.array([e0, e1, e2]).T
        I = np.eye(3)
        B0 = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0]])
        B1 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

        x = np.empty(e.shape)
        x[0] = e[0]
        for t in range(1, 300):
            x[t] = np.linalg.solve(I-B0, np.dot(B1, x[t-1])) + np.linalg.solve(I-B0, e[t])
        X = pd.DataFrame({'x0': x[:, 0], 'x1': x[:, 1], 'x2': x[:, 2]})
        #X = pd.DataFrame({'x0': x[:, 2], 'x1': x[:, 1], 'x2': x[:, 0]})
        #X = pd.DataFrame({'x0': x[:, 1], 'x1': x[:, 0], 'x2': x[:, 2]})
        model = cm.SVARDirectLiNGAM(X.values, X.columns)
        result = model.fit(regression="lasso", lag=1, alpha=0.1)
        result.plot(decimal=3)
        print(result.instantaneous_order)
        print(result.matrixes)
        self.r = result

    def test_instantaneous_order(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.instantaneous_order,
                np.array([0, 1, 2])))

if __name__ == '__main__':
    unittest.main()
