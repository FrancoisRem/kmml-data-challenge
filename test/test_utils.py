from unittest import TestCase

import numpy as np
from numpy import testing

from utils import *


class Test(TestCase):
    def test_binary_regression_labels(self):
        testing.assert_equal(
            binary_regression_labels(np.array([0, 0, 1])),
            np.array([-1, -1, 1]))

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

    def test_standardize_train_test(self):
        X_train = np.vstack([-3, 0])
        X_test = np.vstack([1, 0, -1, -10, 5])

        X_train, X_test = standardize_train_test(X_train, X_test)
        testing.assert_almost_equal(X_train, np.array([[-1.],
                                                       [1.]]))
        testing.assert_almost_equal(X_test, np.array([[1.66666667],
                                                      [1.],
                                                      [0.33333333],
                                                      [-5.66666667],
                                                      [4.33333333]]))

        X_train = np.array([[0, 0], [0, 0], [1, 2], [1, 2]])
        X_test = np.array([[1, 10], [5, 0]])

        X_train, X_test = standardize_train_test(X_train, X_test)
        testing.assert_almost_equal(X_train, np.array([[-1., -1.],
                                                       [-1., -1.],
                                                       [1., 1.],
                                                       [1., 1.]]))
        testing.assert_almost_equal(X_test, np.array([[1., 9.],
                                                      [9., -1.]]))
