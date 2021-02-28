from unittest import TestCase

import numpy as np
from numpy import testing

from utils import binary_regression_labels, linear_kernel_gram_matrix


class Test(TestCase):
    def test_binary_regression_labels(self):
        testing.assert_equal(
            binary_regression_labels(np.array([0, 0, 1])),
            np.array([-1, -1, 1]))

    def test_linear_kernel_gram_matrix_1d(self):
        X = np.vstack([2])
        Y = np.vstack([3])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), 6)

        X = np.vstack([7])
        Y = np.vstack([0])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), 0)

    def test_linear_kernel_gram_matrix_2d(self):
        X = np.array([[1, 2], [6, 6]])
        Y = np.array([[0, 1], [2, 1]])
        expected = np.array([[2, 4], [6, 18]])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), expected)

        X = np.array([[1, 0]])
        Y = np.array([[0, 1], [2, 1]])
        expected = np.array([[0, 2]])
        testing.assert_almost_equal(linear_kernel_gram_matrix(X, Y), expected)
