from unittest import TestCase

import numpy as np
from numpy import testing

from models import KernelModel, KernelRidgeClassifier


class TestKernelModel(TestCase):
    def test__gram_matrix(self):
        X = np.array([[1, 2], [-3, 5], [6, 6], [-10, 5], [7, 0]])
        testing.assert_almost_equal(KernelModel()._gram_matrix(X, X),
                                    np.array([[5., 7., 18., 0., 7.],
                                              [7., 34., 12., 55., -21.],
                                              [18., 12., 72., -30., 42.],
                                              [0., 55., -30., 125., -70.],
                                              [7., -21., 42., -70., 49.]]))


class TestKernelRidgeClassifier(TestCase):
    def setUp(self) -> None:
        self.model = KernelRidgeClassifier(alpha=1.0)
        self.X = np.array([[1, 2], [-3, 5], [6, 6], [-10, 5], [7, 0]])
        self.y = np.array([1, 1, 0, 1, 0])

    def test_fit(self):
        self.assertIsNone(self.model.X_fit_)
        self.assertIsNone(self.model.dual_coef_)

        self.assertIsNotNone(self.model.fit(self.X, self.y))
        testing.assert_equal(self.model.X_fit_, self.X)
        # obtained using cls = sklearn.kernel_ridge.KernelRidge(alpha=n=5)
        # and cls.fit(X, 2*y-1)
        expected_dual_coef = np.array(
            [0.21273056, 0.0985934, -0.08913579, -0.07097586, -0.03043074])
        testing.assert_almost_equal(self.model.dual_coef_, expected_dual_coef)

    def test_fit_with_sample_weights(self):
        self.assertIsNone(self.model.X_fit_)
        self.assertIsNone(self.model.dual_coef_)

        sample_weight = np.array([.3, 2, 9, 7, 1.2])
        self.assertIsNotNone(
            self.model.fit(self.X, self.y, sample_weight=sample_weight))
        testing.assert_equal(self.model.X_fit_, self.X)
        # obtained using cls = sklearn.kernel_ridge.KernelRidge(alpha=n=5)
        # and cls.fit(X, 2*y-1, sample_weight=sample_weight)
        expected_dual_coef = np.array(
            [0.0709017, 0.303963, -0.13714155, -0.17329913, -0.02791378])
        testing.assert_almost_equal(self.model.dual_coef_, expected_dual_coef)

    def test_decision_function(self):
        self.model.X_fit_ = np.vstack([1, 2])
        self.model.dual_coef_ = np.array([3, 5])
        testing.assert_almost_equal(
            self.model.decision_function(np.vstack([1, -1])),
            np.array(
                [3 * 1 + 5 * 2, 3 * (-1) + 5 * (-2)]))

    def test_predict(self):
        self.model.X_fit_ = np.vstack([1, 2])
        self.model.dual_coef_ = np.array([3, 5])
        testing.assert_almost_equal(
            self.model.predict(np.vstack([1, -1])),
            np.array([1, 0]))
