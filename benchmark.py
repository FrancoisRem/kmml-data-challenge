import random
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import GridSearchCV

from kmer_processor import *
from models import *

# Path prefix constants
DATA_FILE_PREFIX = "data/"
TRAINING_FILE_PREFIX = "Xtr"
LABEL_FILE_PREFIX = "Ytr"
FEATURE_FILE_PREFIX = "features/"


def read_train_dataset(k):
    Xtr_df = pd.read_csv(
        DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + ".csv")
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    return pd.merge(left=Xtr_df, right=Ytr_df, on='Id')


def random_splitting(full_matrix_features, full_label_vector, test_size=0.20):
    ### Split Train and Validation
    test_idx = random.sample(range(0, 2000),
                             int(full_label_vector.shape[0] * test_size))
    train_idx = [i for i in range(0, 2000) if not (i in test_idx)]

    X_train = full_matrix_features[train_idx, :]
    y_train = full_label_vector[train_idx]

    X_test = full_matrix_features[test_idx, :]
    y_test = full_label_vector[test_idx]

    return X_train, X_test, y_train, y_test


# %% SELECT FEATURES
kmer_size = 9
number_misplacements = 1
test_size = 0.25
scaling_features = False

use_sparse_kmer_process = False
do_cross_val_grid_search = False
cross_val_kfold_k = 5

# Models to benchmark Train/Test evaluation.
MODELS = [
    KernelSVMClassifier(kernel=LINEAR_KERNEL, alpha=1e-4),
]

# Model and parameters to benchmark using cross-validation grid-search.
CV_MODEL = KernelSVMClassifier()
CV_TUNED_PARAMS = [
    {'kernel': [GAUSSIAN_KERNEL], 'alpha': [5 * 1e-5, 1e-5, 5 * 1e-6, 1e-6]}]

# %% RUN FULL PIPELINE
for k in range(3):
    # Reinitialize models at each iteration
    models = deepcopy(MODELS)
    print(f"------PREDICTION FILE {k}------")
    df = read_train_dataset(k)

    if use_sparse_kmer_process:
        processor = SparseKMerProcessor(df['seq'])
        spectrums = processor.compute_kmer_mismatch(kmer_size,
                                                    number_misplacements)

        spectrums_matrix = compute_spectrums_matrix(spectrums,
                                                    processor.kmers_support,
                                                    sparse=False)
    else:
        processor = DenseKMerProcessor(df['seq'])
        spectrums = processor.compute_kmer_mismatch(kmer_size,
                                                    number_misplacements)
        spectrums_matrix = compute_spectrums_matrix(spectrums,
                                                    sparse=True)

    if issparse(spectrums_matrix):
        size_bytes = spectrums_matrix.data.nbytes \
                     + spectrums_matrix.indptr.nbytes \
                     + spectrums_matrix.indices.nbytes
    else:
        size_bytes = spectrums_matrix.nbytes
    print(
        f"Spectrums_matrix shape: {spectrums_matrix.shape}, size: {size_bytes / 2 ** 30:.2f}Gb")

    # a = spectrums_matrix
    # if type(a) == np.array:
    #     pass
    # else:
    # print(
    #     f"size {(a.data.nbytes + a.indptr.nbytes + a.indices.nbytes) / 10 ** 9}")
    # exit()

    # np.save(FEATURE_FILE_PREFIX + train_name_features, spectrums_matrix)

    X_train, X_test, y_train, y_test = random_splitting(spectrums_matrix,
                                                        df[
                                                            'Bound'].to_numpy(),
                                                        test_size)

    if scaling_features:
        X_train, X_test = standardize_train_test(X_train, X_test)

    # Cross-validation-based grid-search for CV_MODEL over CV_TUNED_PARAMS.
    if do_cross_val_grid_search:
        clf = GridSearchCV(
            CV_MODEL, CV_TUNED_PARAMS, scoring='accuracy',
            n_jobs=3, cv=cross_val_kfold_k
        )
        print(f"Starting grid-search cross-validation:\n\t{clf}\n")
        clf.fit(X_train, y_train)

        print(
            f"Best parameters set found on development set: {clf.best_params_}")
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(f"\t{mean * 100:.1f}% (+/-{std * 2 * 100:.1f}%) for {params}")
        print()

        # Final evaluation training on the whole train dataset et evaluating
        # on the unseen test dataset with the best found parameters.
        model = clf.best_estimator_
        print(
            f"Train and evaluate best model: {model.__class__.__name__} "
            f"{model.get_params()}")
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train_pred, y_train)

        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test_pred, y_test)

        print(
            f"\tTrain {train_accuracy * 100:.1f}% "
            f"| Test:{test_accuracy * 100:.1f}%\n")

    # Classic Train/Test comparison over MODELS list.
    else:
        for model in models:
            print(
                f"Train and evaluate model: {model.__class__.__name__} "
                f"{model.__dict__}")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train_pred, y_train)

            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test_pred, y_test)

            print(
                f"\tTrain {train_accuracy * 100:.1f}% "
                f"| Test:{test_accuracy * 100:.1f}%")
