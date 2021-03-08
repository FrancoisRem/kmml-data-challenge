from unittest import TestCase

import numpy as np
from numpy import testing

from utils import accuracy_score, binary_regression_labels, \
    gaussian_kernel_gram_matrix, \
    linear_kernel_gram_matrix


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

    def test_gaussian_kernel_gram_matrix_1d(self):
        X = np.vstack([2])
        Y = np.vstack([3])
        # expected: exp(-1*||3-2||^2)=exp(-1)~0.3678794411
        testing.assert_almost_equal(gaussian_kernel_gram_matrix(X, Y, gamma=1),
                                    0.3678794411)

        X = np.vstack([7])
        Y = np.vstack([0])
        # expected: exp(-0.05*||7-0||^2)=exp(-2.45)~0.08629358649
        testing.assert_almost_equal(
            gaussian_kernel_gram_matrix(X, Y, gamma=0.05),
            0.08629358649)

    def test_gaussian_kernel_gram_matrix_2d(self):
        X = np.array([[1, 2], [6, 6]])
        Y = np.array([[0, 1], [2, 1]])

        # expected = np.array([[exp(-2), exp(-2)], [exp(-61), exp(-41)]])
        expected = np.array(
            [[0.13533528323, 0.13533528323], [3.2213403e-27, 1.5628822e-18]])
        testing.assert_almost_equal(gaussian_kernel_gram_matrix(X, Y, gamma=1),
                                    expected)

        X = np.array([[1, 0]])
        Y = np.array([[0, 1], [2, 1]])
        # expected = [[exp(-2), exp(-2)]]
        expected = np.array([[0.13533528323, 0.13533528323]])
        testing.assert_almost_equal(gaussian_kernel_gram_matrix(X, Y, gamma=1),
                                    expected)

    def test_accuracy_score(self):
        self.assertAlmostEqual(
            accuracy_score(np.array([1, 0, 1]), np.array([1, 0, 0])),
            2 / 3)
        self.assertAlmostEqual(
            accuracy_score(np.array([1, 0, 0]), np.array([1, 0, 0])),
            1)
        self.assertAlmostEqual(
            accuracy_score(np.array([1, 0]), np.array([0, 1])),
            0)
        self.assertAlmostEqual(
            accuracy_score(np.array([3, 5, 5]), np.array([5, 5, 3])),
            1 / 3)
