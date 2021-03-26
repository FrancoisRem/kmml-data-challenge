import numpy as np
from scipy import sparse

def linear_kernel_gram_matrix(X, Y):
    """
    Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
    using the linear kernel K(x, y) = <x, y>.
    This function does not assert the dimensions of X and Y!
    :param X: np.array, or scipy.sparse matrix, with shape nX, d
    :param Y: np.array, or scipy.sparse matrix, with shape nY, d
    :return: np.array with shape nX, nY
    """
    if sparse.issparse(X):
        return (X @ Y.T).toarray()
    return X @ Y.T


def gaussian_kernel_gram_matrix(X, Y, gamma):
    """
    Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
    using the linear kernel K(x, y) = exp(-gamma * ||x - y||^2).
    This function does not assert the dimensions of X and Y!
    :param X: np.array, or scipy.sparse matrix, with shape nX, d
    :param Y: np.array, or scipy.sparse matrix, with shape nY, d
    :param gamma: float, gamma coefficient in the exponential
    :return: np.array with shape nX, nY
    """
    if gamma == 'auto':
        gamma = 1 / X.shape[1]
    if sparse.issparse(X):
        sqnorm_X = sparse.linalg.norm(X, axis=1) ** 2
        sqnorm_Y = sparse.linalg.norm(Y, axis=1) ** 2
        distances = sqnorm_X[..., np.newaxis] + sqnorm_Y - 2 * (
                X @ Y.T).toarray()
    else:
        sqnorm_X = np.linalg.norm(X, axis=1) ** 2
        sqnorm_Y = np.linalg.norm(Y, axis=1) ** 2
        distances = sqnorm_X[..., np.newaxis] + sqnorm_Y - 2 * X @ Y.T
    return np.exp(-gamma * distances)


def cosine_similarity_kernel_gram_matrix(X, Y):
    """
    Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
    using the cosine similarity (or normalized linear) kernel
    K(x, y) = <x, y>/(||x||*||y||).
    This function expect all the rows of X and Y to be non-identically zero.
    This function does not assert the dimensions of X and Y!
    :param X: np.array, or scipy.sparse matrix, with shape nX, d
    :param Y: np.array, or scipy.sparse matrix, with shape nY, d
    :return: np.array with shape nX, nY
    """
    if np.all(X == Y):
        linear = linear_kernel_gram_matrix(X, X)
        inv_sqrt_diag = np.sqrt(1 / linear.diagonal())
        return inv_sqrt_diag * linear * np.vstack(inv_sqrt_diag)

    # sparse matrices not supported yet.
    linear = linear_kernel_gram_matrix(X, Y)
    inv_sqrt_X = 1 / np.linalg.norm(X, axis=1)
    inv_sqrt_Y = 1 / np.linalg.norm(Y, axis=1)
    return np.vstack(inv_sqrt_X) * linear * inv_sqrt_Y
