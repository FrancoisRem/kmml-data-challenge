"""
The algorithms to optimize the models are inspired by the Kernel methods for
machine learning class by Jean-Philippe Vert and Julien Mairal
(http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2021mva/index.html).
The implementation of the models is inspired by the scikit-learn library
(https://scikit-learn.org/stable/).
"""

import numpy as np

from scipy import linalg

from utils import *


def linear_kernel(x, y):
    return np.dot(x, y)


class KernelModel:
    """
    Base class for models using a Kernel function.
    """

    def __init__(self, kernel=linear_kernel):
        """
        :param kernel: callable function that takes two n,d np.arrays and
        returns a number. This function should be a positive definite Kernel
        (https://en.wikipedia.org/wiki/Positive-definite_kernel).
        """
        assert callable(kernel)
        self.kernel_ = kernel

    def _gram_matrix(self, X, Y):
        """
        Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
        using kernel_.
        :param X: np.array with shape nX, d
        :param Y: np.array with shape nY, d
        :return: np.array with shape nX, nY
        """
        nX, dX = X.shape
        nY, dY = Y.shape
        assert dX == dY
        K = np.zeros([nX, nY])
        for i in range(nX):
            for j in range(nY):
                K[i, j] = self.kernel_(X[i], Y[j])
        return K


class KernelRidgeClassifier(KernelModel):
    """
    Binary Ridge classifier model using kernel methods. This classifier
    first converts the target values into {-1, 1} and then treats the
    problem as a regression task.
    """

    def __init__(self, alpha=1, kernel=linear_kernel):
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

    def decision_function(self, X):
        K = self._gram_matrix(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)

    def predict(self, X):
        scores = self.decision_function(X)
        return np.asarray((scores > 0).astype(int))
