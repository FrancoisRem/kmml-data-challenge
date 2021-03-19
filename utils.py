import numpy as np


def binary_regression_labels(y):
    # transform {0, 1} labels to {-1, 1} labels
    # (https://scikit-learn.org/stable/modules/linear_model.html#classification)
    return 2 * y - 1


def linear_kernel_gram_matrix(X, Y):
    """
    Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
    using the linear kernel K(x, y) = <x, y>.
    This function does not assert the dimensions of X and Y!
    :param X: np.array with shape nX, d
    :param Y: np.array with shape nY, d
    :return: np.array with shape nX, nY
    """
    return X @ Y.T


def gaussian_kernel_gram_matrix(X, Y, gamma):
    """
    Compute the (Kernel) Gram matrix between X and Y: K_{i,j} = K(X_i, Y_j)
    using the linear kernel K(x, y) = exp(-gamma * ||x - y||^2).
    This function does not assert the dimensions of X and Y!
    :param X: np.array with shape nX, d
    :param Y: np.array with shape nY, d
    :param gamma: float, gamma coefficient in the exponential
    :return: np.array with shape nX, nY
    """
    if gamma == 'auto':
        gamma = 1 / X.shape[1]
    sqnorm_X = np.linalg.norm(X, axis=1) ** 2
    sqnorm_Y = np.linalg.norm(Y, axis=1) ** 2
    distances = sqnorm_X[..., np.newaxis] + sqnorm_Y - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * distances)


def accuracy_score(predicted, expected):
    """
    Compute the accuracy score between predicted and expected labels.
    :param predicted: np.array with shape (n,) of two int labels
    :param expected: np.array with shape (n,) of two int labels
    :return: accuracy score in [0, 1]
    """
    n = len(predicted)
    assert n > 0
    assert predicted.shape == expected.shape == (n,)
    return (predicted == expected).sum() / n


def standardize_train_test(X_train, X_test):
    """
    Center and scale X_train and X_test by X_train mean and std along axis.
    :param X_train: (nX, d)-np.array
    :param X_test:  (nY, d)-np.array
    :return: (nX, d)-np.array, (nY, d)-np.array
    """
    X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_test
