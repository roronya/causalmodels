import causalmodels as cm
import numpy as np
import pandas as pd
import unittest

class TestLiNGAMMethods(unittest.TestCase):
    def setUp(self):
        e0 = np.sin(np.arange(0, 30, 0.1))
        e1 = np.random.laplace(size=300)
        e2 = np.random.exponential(size=300)
        e = np.array([e0, e1, e2])
        I = np.eye(3)
        B = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 0]])
        X = np.linalg.solve(I-B, e)
        data = pd.DataFrame(X.T, columns=['x0', 'x1', 'x2'])
        model = cm.DirectLiNGAM(data.values, data.columns)
        result = model.fit(regression="lasso")
        result.plot(decimal=3)
        print(result.order)
        print(result.matrix)
        self.r = result

    def test_order(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.order,
                np.array([0, 1, 2])))

if __name__ == '__main__':
    unittest.main()
