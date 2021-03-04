"""
The algorithms to optimize the models are inspired by the Kernel methods for
machine learning class by Jean-Philippe Vert and Julien Mairal
(http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2021mva/index.html).
The implementation of the models is inspired by the scikit-learn library
(https://scikit-learn.org/stable/).
"""

import numpy as np
import cvxpy as cp

from scipy import linalg
from scipy.special import expit

from utils import *

# Constant for linear kernel K(x, y) = <x, y>
LINEAR_KERNEL = 'lin'
# Constant for Gaussian (Radial Basis Function) kernel
# K(x, y) = exp(-gamma * ||x - y||^2)
GAUSSIAN_KERNEL = 'rbf'


class KernelModel:
    """
    Base class for models using a Kernel function.
    """

    def __init__(self, kernel=LINEAR_KERNEL, gamma=1):
        """
        :param kernel: LINEAR_KERNEL,
        or callable function that takes two n,d np.arrays and
        returns a number. This function should be a positive definite Kernel
        (https://en.wikipedia.org/wiki/Positive-definite_kernel).
        :param gamma: float, coefficient for Gaussian kernel
        """
        assert kernel in [LINEAR_KERNEL, GAUSSIAN_KERNEL] or callable(kernel)
        self.kernel_ = kernel
        self.gamma_ = gamma

    def _gram_matrix(self, X, Y):
        """
        Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
        using kernel_.
        :param X: np.array with shape nX, d
        :param Y: np.array with shape nY, d
        :return: np.array with shape nX, nY
        """
        if self.kernel_ == LINEAR_KERNEL:
            return linear_kernel_gram_matrix(X, Y)

        if self.kernel_ == GAUSSIAN_KERNEL:
            return gaussian_kernel_gram_matrix(X, Y, self.gamma_)

        nX, dX = X.shape
        nY, dY = Y.shape
        assert dX == dY
        K = np.zeros([nX, nY])
        for i in range(nX):
            for j in range(nY):
                K[i, j] = self.kernel_(X[i], Y[j])
        return K


class LinearKernelBinaryClassifier(KernelModel):
    """
    Base class for Kernel Binary Classifiers using a linear decision_function
    which is the weighted sum of the kernel product between learnt vectors
    X_fits[i] and query vectors X[i], the weights are contained in the learnt
    vector dual_coef_.
    """

    def __init__(self, kernel=LINEAR_KERNEL, gamma=1):
        """
        :param kernel: see KernelModel doc
        """
        self.X_fit_ = None
        self.dual_coef_ = None
        super().__init__(kernel=kernel)

    def decision_function(self, X):
        self.assert_is_fitted()
        K = self._gram_matrix(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)

    def predict(self, X):
        self.assert_is_fitted()
        scores = self.decision_function(X)
        return np.asarray((scores > 0).astype(int))

    def assert_is_fitted(self):
        if self.X_fit_ is None or self.dual_coef_ is None:
            raise ValueError(f"{self.__class__.__name__} is not fitted.")


class KernelRidgeClassifier(LinearKernelBinaryClassifier):
    """
    Binary Ridge classifier model using kernel methods. This classifier
    first converts the target values into {-1, 1} and then treats the
    problem as a regression task.
    """

    def __init__(self, alpha=1, kernel=LINEAR_KERNEL, gamma=1):
        """
        :param alpha: L2 regularization weight, must be a positive float
        :param kernel: see KernelModel doc
        """
        self.alpha_ = alpha
        self.dual_coef_ = None
        self.X_fit_ = None
        super().__init__(kernel=kernel)

    def fit(self, X, y, sample_weight=None):
        y = binary_regression_labels(y)
        K = self._gram_matrix(X, X)
        n, _ = X.shape
        if sample_weight is None:
            self.dual_coef_ = linalg.solve(
                a=K + n * self.alpha_ * np.eye(n),
                b=y,
                assume_a='pos')
        else:
            # Weighted Kernel Ridge Regression
            assert n == sample_weight.shape[0]
            assert np.all(sample_weight >= 0)
            Wsqrt = np.diag(np.sqrt(sample_weight))
            self.dual_coef_ = Wsqrt @ linalg.solve(
                a=Wsqrt @ K @ Wsqrt + n * self.alpha_ * np.eye(n),
                b=Wsqrt.dot(y),
                assume_a='pos')
        self.X_fit_ = X
        return self


class KernelLogisticClassifier(LinearKernelBinaryClassifier):
    """
    Binary Logistic classifier model using kernel methods. This classifier
    first converts the target values into {-1, 1} and then treats the
    problem as a regression task.
    """

    def __init__(self, alpha=1, kernel=LINEAR_KERNEL,gamma=1):
        self.alpha_ = alpha
        self.dual_coef_ = None
        self.X_fit_ = None
        super().__init__(kernel=kernel)

    def fit(self, X, y, tol=1e-4, max_iter=100):
        """
        Compute self.dual_coef c to minimize:
        1/n sum_i^n log(1 + exp(-y_i * K@c_i)) + alpha / 2 * c.T @ K @ c.
        """
        y = binary_regression_labels(y)
        K = self._gram_matrix(X, X)
        n, _ = X.shape

        coef = np.zeros(n)
        for it in range(max_iter):

            # Update variables
            prev_coef = coef
            m = K @ coef
            P = -expit(- y * m)
            W = expit(y * m) * expit(-y * m)
            z = m - P * y / W

            # Solve Weighted Kernel Ridge Regression problem
            Wsqrt = np.diag(np.sqrt(W))
            coef = Wsqrt @ linalg.solve(
                a=Wsqrt @ K @ Wsqrt + n * self.alpha_ * np.eye(n),
                b=Wsqrt.dot(z),
                assume_a='pos')

            if np.linalg.norm(prev_coef - coef) < tol:
                break

            if it == max_iter - 1:
                print(
                    f"Maximum iteration {max_iter} reached in "
                    f"{self.__class__.__name__} fit method.")

        self.dual_coef_ = coef
        self.X_fit_ = X
        return self


class KernelSVMClassifier(LinearKernelBinaryClassifier):
    """
    Binary SVM classifier model using kernel methods.
    """

    def __init__(self, alpha=1, kernel=LINEAR_KERNEL, gamma=1):
        self.alpha_ = alpha
        self.dual_coef_ = None
        self.X_fit_ = None
        super().__init__(kernel=kernel)

    def fit(self, X, y, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000):
        """
        Compute self.dual_coef c to minimize: 2 * y.T @ c - c.T @ K @ c
        s.t. 0 <= c_i * y_i <= 1 / (2 * alpha_ * n) for all i
        The solver used is OSQP through the cvxpy library.
        :param X: np.array with shape n, d
        :param y: np.array with shape (n,)
        :param eps_abs: absolute accuracy (OSQP solver parameter)
        :param eps_rel: relative accuracy (OSQP solver parameter)
        :param max_iter: maximum number of iterations (OSQP solver parameter)
        :return: self
        """
        y = binary_regression_labels(y)
        K = self._gram_matrix(X, X)
        n, _ = X.shape

        coef = cp.Variable(n)
        problem = cp.Problem(
            cp.Maximize(2 * y.T @ coef - cp.quad_form(coef, K)),
            [0 <= cp.multiply(y, coef),
             cp.multiply(y, coef) <= 1 / (2 * self.alpha_ * n)])
        problem.solve(solver='OSQP', max_iter=max_iter, eps_abs=eps_abs,
                      eps_rel=eps_rel)
        if coef.value is None:
            raise Exception(
                f"Solver fot {self.__class__.__name__} fit method failed with"
                f" status {problem.status}")

        self.X_fit_ = X
        self.dual_coef_ = coef.value
        return self
