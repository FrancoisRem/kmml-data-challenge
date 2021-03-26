import random
from copy import deepcopy

import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV

from kmer_processor import *
from models import *
from utils import *

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


def random_splitting(full_matrix_features, full_label_vector, test_size=0.20,
                     test_idx=None):
    ### Split Train and Validation
    if test_idx is None:
        test_idx = random.sample(range(0, 2000),
                                 int(full_label_vector.shape[0] * test_size))
    train_idx = [i for i in range(0, 2000) if not (i in test_idx)]

    X_train = full_matrix_features[train_idx, :]
    y_train = full_label_vector[train_idx]

    X_test = full_matrix_features[test_idx, :]
    y_test = full_label_vector[test_idx]

    return X_train, X_test, y_train, y_test


def process_kmer_dataset(df, kmer_size, number_misplacements, test_size=0.20,
                         test_idx=None, scaling_features=True,
                         exhaustive_spectrum=True, use_sparse_matrix=True):
    if exhaustive_spectrum:
        processor = DenseKMerProcessor(df['seq'])
        spectrums = processor.compute_kmer_mismatch(kmer_size,
                                                    number_misplacements)
        spectrums_matrix = compute_spectrums_matrix(spectrums,
                                                    sparse=use_sparse_matrix)
    else:
        processor = SparseKMerProcessor(df['seq'])
        spectrums = processor.compute_kmer_mismatch(kmer_size,
                                                    number_misplacements)

        spectrums_matrix = compute_spectrums_matrix(spectrums,
                                                    processor.kmers_support,
                                                    sparse=use_sparse_matrix)

    if issparse(spectrums_matrix):
        size_bytes = spectrums_matrix.data.nbytes \
                     + spectrums_matrix.indptr.nbytes \
                     + spectrums_matrix.indices.nbytes
    else:
        size_bytes = spectrums_matrix.nbytes
    print(
        f"Spectrums_matrix shape: {spectrums_matrix.shape},"
        f" size: {size_bytes / 2 ** 30:.2f}Gb")

    X_train, X_test, y_train, y_test = random_splitting(spectrums_matrix,
                                                        df['Bound'].to_numpy(),
                                                        test_size=test_size,
                                                        test_idx=test_idx)

    if scaling_features:
        X_train, X_test = standardize_train_test(X_train, X_test)

    return X_train, X_test, y_train, y_test


# %% SELECT FEATURES
kmer_size = 7
number_misplacements = 1
test_size = 0.25
scaling_features = False

use_sparse_matrix = False
exhaustive_spectrum = True
do_cross_val_grid_search = False
cross_val_kfold_k = 5

# If not empty, --> use sum kernel: provide list of kernel as kernel with the
# same size as SUM_KERNEL_PARAMS.
SUM_KERNEL_SPECTRUM_PARAMS = [(5, 1), (7, 1)]
SUM_KERNEL_KERNELS = [(COSINE_KERNEL, 0.1), (COSINE_KERNEL, 0.9)]

# Models to benchmark Train/Test evaluation.
MODELS = [
    KernelSVMClassifier(kernel=SUM_KERNEL_KERNELS, C=10),
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

    if SUM_KERNEL_SPECTRUM_PARAMS:
        df = read_train_dataset(k)
        X_train_list = []
        X_test_list = []
        full_label_vector = df['Bound'].to_numpy()

        # Very important: need consistent indices for train/test split.
        test_idx_sum_kernel = random.sample(range(0, 2000),
                                            int(full_label_vector.shape[
                                                    0] * test_size))

        for km_size, nb_mismatch in SUM_KERNEL_SPECTRUM_PARAMS:
            X_train, X_test, y_train, y_test = process_kmer_dataset(df,
                                                                    km_size,
                                                                    nb_mismatch,
                                                                    test_size=test_size,
                                                                    test_idx=test_idx_sum_kernel,
                                                                    scaling_features=scaling_features,
                                                                    exhaustive_spectrum=exhaustive_spectrum,
                                                                    use_sparse_matrix=use_sparse_matrix)
            X_train_list.append(X_train)
            X_test_list.append(X_test)

        X_train = X_train_list
        X_test = X_test_list

    else:
        X_train, X_test, y_train, y_test = process_kmer_dataset(df, kmer_size,
                                                                number_misplacements,
                                                                test_size=test_size,
                                                                test_idx=None,
                                                                scaling_features=scaling_features,
                                                                exhaustive_spectrum=exhaustive_spectrum,
                                                                use_sparse_matrix=use_sparse_matrix)

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
