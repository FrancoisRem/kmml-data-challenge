from unittest import TestCase

import numpy as np
from numpy import testing

from models import KernelLogisticClassifier, KernelModel, KernelRidgeClassifier, \
    KernelSVMClassifier, LinearKernelBinaryClassifier


class TestKernelModel(TestCase):
    def test__gram_matrix(self):
        X = np.array([[1, 2], [-3, 5], [6, 6], [-10, 5], [7, 0]])
        testing.assert_almost_equal(KernelModel()._gram_matrix(X, X),
                                    np.array([[5., 7., 18., 0., 7.],
                                              [7., 34., 12., 55., -21.],
                                              [18., 12., 72., -30., 42.],
                                              [0., 55., -30., 125., -70.],
                                              [7., -21., 42., -70., 49.]]))


class TestLinearKernelBinaryClassifier(TestCase):
    def setUp(self):
        self.model = LinearKernelBinaryClassifier()

    def test_assert_is_fitted_not_fitted(self):
        self.assertRaises(ValueError, self.model.assert_is_fitted)

    def test_assert_is_fitted_is_fitted(self):
        self.model.X_fit_ = 0
        self.model.dual_coef_ = 0
        try:
            self.model.assert_is_fitted()
        except ValueError:
            self.fail("assert_is_fitted() raised ValueError unexpectedly!")

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


class TestKernelLogisticClassifier(TestCase):
    def setUp(self):
        self.model = KernelLogisticClassifier(alpha=2)

    def test_fit_1d(self):
        X = np.vstack([1])
        y = np.array([1])

        self.assertIsNone(self.model.X_fit_)
        self.assertIsNone(self.model.dual_coef_)

        self.assertIsNotNone(self.model.fit(X, y))
        testing.assert_equal(self.model.X_fit_, X)

        # obtained using WolframAlpha
        expected_dual_coef = 0.2223234712
        testing.assert_almost_equal(self.model.dual_coef_,
                                    expected_dual_coef)

    def test_fit_2d(self):
        # We make sure that X is such that X @ X.T is spd such that the problem
        # is stritly convex and there is a unique solution.
        X = np.array([[1, 2], [6, 6]])
        y = np.array([0, 1])

        self.assertIsNone(self.model.X_fit_)
        self.assertIsNone(self.model.dual_coef_)

        self.assertIsNotNone(self.model.fit(X, y))
        testing.assert_equal(self.model.X_fit_, X)

        # obtained using cvxpy in script.py
        expected_dual_coef = np.array([-0.14089595, 0.05334446])
        testing.assert_almost_equal(self.model.dual_coef_, expected_dual_coef)


class TestKernelSVMClassifier(TestCase):
    def test_fit_no_intercept_(self):
        X = np.vstack([1])
        y = np.array([1])

        # The model maximizes f(c)=2*c-c^2 s.t. 0<=c<=1/(2*alpha).
        # f is increasing from -inf to 1 and decreasing from 1 to +inf
        # so the optimal c is min(1, 1/(2*alpha)).
        model = KernelSVMClassifier(alpha=1, intercept=False)
        self.assertIsNone(model.X_fit_)
        self.assertIsNone(model.dual_coef_)

        self.assertIsNotNone(model.fit(X, y))
        testing.assert_equal(model.X_fit_, X)

        expected_dual_coef = 0.5
        testing.assert_almost_equal(model.dual_coef_,
                                    expected_dual_coef)

        model = KernelSVMClassifier(alpha=0.5, intercept=False).fit(X, y)
        expected_dual_coef = 1
        testing.assert_almost_equal(model.dual_coef_,
                                    expected_dual_coef)

        model = KernelSVMClassifier(alpha=0.25, intercept=False).fit(X, y)
        expected_dual_coef = 1
        testing.assert_almost_equal(model.dual_coef_,
                                    expected_dual_coef)

    def test_fit_intercept_1d(self):
        X = np.vstack([1, 3])
        y = np.array([0, 1])
        # Small alpha value not to over-regularize for this test.
        model = KernelSVMClassifier(intercept=True, alpha=1e-5)
        self.assertIsNone(model.X_fit_)
        self.assertIsNone(model.dual_coef_)

        self.assertIsNotNone(model.fit(X, y))
        testing.assert_equal(model.X_fit_, X)
        # The intercept is equal to the opposite of the mean of the data in
        # this case.
        np.testing.assert_equal(model.intercept_, -2)
        np.testing.assert_equal(model.predict(X), y)

        # Check problem constraints:
        C = 1 / (2 * X.shape[1] * model.alpha_)
        np.testing.assert_almost_equal(np.sum(model.dual_coef_), 0)
        np.testing.assert_equal(model.dual_coef_ * y >= 0, True)
        np.testing.assert_equal(model.dual_coef_ * y <= C, True)

        # Assert that the same model with intercept=False cannot classify
        # correctly the data.
        model = KernelSVMClassifier(intercept=False).fit(X, y)
        self.assertFalse(np.all(model.predict(X) == y))

    def test_fit_intercept_2d(self):
        X = np.vstack([[0, 1], [1, 0], [2, 2], [4, 0]])
        y = np.array([0, 0, 1, 1])

        # This little classification task can be handled perfectly with a linear
        # SVM which implements an intercept (a bias): for example the affine
        # hyperplan drawn with the symbol '\' correctly separates the data.
        # However the data is not linearly separable without an intercept, so
        # the same linear-SVM without intercept cannot correctly classify the
        # data.
        #
        #     ^
        #  \  |
        #    \|
        #     |\......O
        #     |  \    .
        #     1    \  .
        #     |      \.
        #     @---1---.\------O---->
        #                \
        #
        model = KernelSVMClassifier(intercept=True)
        self.assertIsNone(model.X_fit_)
        self.assertIsNone(model.dual_coef_)

        self.assertIsNotNone(model.fit(X, y))
        testing.assert_equal(model.X_fit_, X)
        np.testing.assert_equal(model.predict(X), y)

        # Check problem constraints:
        C = 1 / (2 * X.shape[1] * model.alpha_)
        np.testing.assert_almost_equal(np.sum(model.dual_coef_), 0)
        np.testing.assert_equal(model.dual_coef_ * y >= 0, True)
        np.testing.assert_equal(model.dual_coef_ * y <= C, True)

        # Assert that the same model with intercept=False cannot classify
        # correctly the data.
        model = KernelSVMClassifier(intercept=False).fit(X, y)
        self.assertFalse(np.all(model.predict(X) == y))
