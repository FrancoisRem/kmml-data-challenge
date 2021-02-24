import numpy as np


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
