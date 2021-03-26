def binary_regression_labels(y):
    # transform {0, 1} labels to {-1, 1} labels
    # (https://scikit-learn.org/stable/modules/linear_model.html#classification)
    return 2 * y - 1


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
