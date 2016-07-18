import numpy as np
from causalmodels import Result
import unittest

class TestResultMethods(unittest.TestCase):
    def setUp(self):
        order = np.array([2, 0, 1])
        permuted_matrix = np.array([[0, 0, 0],
                                    [1, 0, 0],
                                    [2, 3, 0]])
        data = np.array([[0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2]])
        labels = np.array(['a', 'b', 'c'])
        self.r = Result(order=order, permuted_matrix=permuted_matrix, data=data, labels=labels)

    def test_order(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.order,
                np.array([2, 0, 1])))

    def test_matrix(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.matrix,
                np.array([[0, 0, 1],
                          [3, 0, 2],
                          [0, 0, 0]])))

    def test_permuted_matrix(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.permuted_matrix,
                np.array([[0, 0, 0],
                          [1, 0, 0],
                          [2, 3, 0]])))

    def test_data(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.data,
                np.array([[0, 1, 2],
                          [0, 1, 2],
                          [0, 1, 2],
                          [0, 1, 2],
                          [0, 1, 2]])))

    def test_permuted_data(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.permuted_data,
                np.array([[2, 0, 1],
                          [2, 0, 1],
                          [2, 0, 1],
                          [2, 0, 1],
                          [2, 0, 1]])))

    def test_labels(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
            self.r.labels,
            np.array(['a', 'b', 'c'])
            ))

    def test_permuted_labels(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.permuted_labels,
                np.array(['c', 'a', 'b'])))

    def test_plot(self):
        self.r.plot()

if __name__ == '__main__':
    unittest.main()
