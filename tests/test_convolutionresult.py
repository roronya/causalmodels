import numpy as np
from causalmodels import ConvolutionResult
import unittest

class TestConvolutionResultMethods(unittest.TestCase):
    def setUp(self):
        instantaneous_order = np.array([2, 0, 1])
        permuted_instantaneous_matrix = np.array([[0, 0, 0],
                                                  [1, 0, 0],
                                                  [2, 3, 0]])
        convolution_matrixes = np.array([
                                         [[0, 0, 1],
                                          [0, 2, 0],
                                          [3, 0, 0]],
                                         [[0, 1, 0],
                                          [0, 2, 0],
                                          [0, 3, 0]]
                                          ])
        data = np.array([[0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2],
                         [0, 1, 2]])
        labels = np.array(['a', 'b', 'c'])
        self.r = ConvolutionResult(instantaneous_order=instantaneous_order,
                                   permuted_instantaneous_matrix=permuted_instantaneous_matrix,
                                   convolution_matrixes=convolution_matrixes,
                                   data=data, labels=labels)

    def test_order(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.instantaneous_order,
                np.array([2, 0, 1])))

    def test_matrixes(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.matrixes,
                np.array([
                          [[0, 0, 1],
                           [3, 0, 2],
                           [0, 0, 0]],
                          [[0, 0, 1],
                           [0, 2, 0],
                           [3, 0, 0]],
                          [[0, 1, 0],
                           [0, 2, 0],
                           [0, 3, 0]]])))

    def test_permuted_matrixes(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                self.r.permuted_matrixes,
                np.array([
                          [[0, 0, 0],
                           [1, 0, 0],
                           [2, 3, 0]],
                          [[0, 3, 0],
                           [1, 0, 0],
                           [0, 0, 2]],
                          [[0, 0, 3],
                           [0, 0, 1],
                           [0, 0, 2]]])))

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
