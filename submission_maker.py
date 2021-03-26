"""
Created on Sun Feb 28 13:58:29 2021

@author: theophanegregoir
"""

# %% Imports

import os

import pandas as pd
from scipy.sparse import issparse

from kmer_processor import *
from models import *
from utils import *

# %% Load data

DATA_FILE_PREFIX = "data/"
FEATURE_FILE_PREFIX = "features/"
TRAINING_FILE_PREFIX = "Xtr"
LABEL_FILE_PREFIX = "Ytr"
TEST_FILE_PREFIX = "Xte"


def read_dataset_train_test(k, use_mat_features=True, use_kmers=True,
                            kmer_min_size=3, kmer_max_size=4,
                            with_misplacement=True, number_misplacements=1,
                            dict_original_pattern_to_misplaced=None,
                            scaled=False):
    df_test = pd.read_csv(DATA_FILE_PREFIX + TEST_FILE_PREFIX + str(k) + ".csv")

    Xtr_df = pd.read_csv(
        DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + ".csv")
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    df_train = pd.merge(left=Xtr_df, right=Ytr_df, on='Id')

    list_features = []

    ### Features created by professors
    if use_mat_features:
        ### train
        Xtr_mat100_df = pd.read_csv(
            DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + "_mat100.csv",
            sep=' ', header=None)
        df_train[
            ['feature_given_' + str(j) for j in range(1, 101)]] = Xtr_mat100_df

        ### test
        Xte_mat100_df = pd.read_csv(
            DATA_FILE_PREFIX + TEST_FILE_PREFIX + str(k) + "_mat100.csv",
            sep=' ', header=None)
        df_test[
            ['feature_given_' + str(j) for j in range(1, 101)]] = Xte_mat100_df

        list_features += ['feature_given_' + str(j) for j in range(1, 101)]

    ### Features of kmers frequency
    if use_kmers:
        ### train
        list_kmer_pattern, df_train, dict_original_pattern_to_misplaced = add_kmer_features(
            kmer_min_size, kmer_max_size, with_misplacement,
            number_misplacements, df_train, dict_original_pattern_to_misplaced)

        ### test
        _, df_test, dict_original_pattern_to_misplaced = add_kmer_features(
            kmer_min_size, kmer_max_size, with_misplacement,
            number_misplacements, df_test, dict_original_pattern_to_misplaced)

        list_features += list_kmer_pattern

    ### Create X_train and y_train
    y_train = df_train['Bound'].to_numpy()

    ### Center data and to numpy
    X_train = df_train[list_features].to_numpy()
    X_test = df_test[list_features].to_numpy()

    if scaled:
        print("SCALING")
        X_train, X_test = standardize_train_test(X_train, X_test)

    return X_train, y_train, X_test, dict_original_pattern_to_misplaced


def load_data_for_submission(train_name_features, test_name_features):
    ### Load Xtrain
    X_train = np.load(FEATURE_FILE_PREFIX + train_name_features)

    ### Load Xtest
    X_test = np.load(FEATURE_FILE_PREFIX + test_name_features)

    ### Load Ytrain
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    y_train = Ytr_df['Bound'].to_numpy()

    return X_train, y_train, X_test


def read_dataset_train_test_fast_kmer_process(k, kmer_size=3,
                                              number_misplacements=1,
                                              scaling_features=False,
                                              exhaustive_spectrum=True,
                                              use_sparse_matrix=False):
    df_test = pd.read_csv(DATA_FILE_PREFIX + TEST_FILE_PREFIX + str(k) + ".csv")

    Xtr_df = pd.read_csv(
        DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + ".csv")
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    df_train = pd.merge(left=Xtr_df, right=Ytr_df, on='Id')

    total_seq_series = pd.concat([df_train['seq'], df_test['seq']],
                                 ignore_index=True)

    # print(type(total_seq_series))

    if exhaustive_spectrum:

        processor = DenseKMerProcessor(total_seq_series)

        spectrums = processor.compute_kmer_mismatch(kmer_size,
                                                    number_misplacements)

        spectrums_matrix = compute_spectrums_matrix(spectrums,
                                                    sparse=use_sparse_matrix)

    else:
        processor = SparseKMerProcessor(total_seq_series)

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

    X_train = spectrums_matrix[:2000, :]
    X_test = spectrums_matrix[2000:, :]

    ### Create X_train and y_train
    y_train = df_train['Bound'].to_numpy()

    if scaling_features:
        X_train, X_test = standardize_train_test(X_train, X_test)

    return X_train, y_train, X_test


# %% Actual Fitting and prediction

### 3 different datasets
test_prediction = {}
dict_original_pattern_to_misplaced = None

kmer_min_size = 8
kmer_max_size = 8
number_misplacements = 2
with_misplacement = True
scaling_features = False

use_fast_kmer_process = True
use_sparse_matrix = True
exhaustive_spectrum = True

# If not empty, --> use sum kernel: provide list of kernel as kernel with the
# same size as SUM_KERNEL_PARAMS.
SUM_KERNEL_SPECTRUM_PARAMS = [(6, 1), (9, 1)]
SUM_KERNEL_KERNELS = [(LINEAR_KERNEL, 1e-2), (LINEAR_KERNEL, 1)]

# Models to benchmark Train/Test evaluation.
MODELS = [
    KernelLogisticClassifier(kernel=SUM_KERNEL_KERNELS, alpha=0.5),
]

# Handle single-model case.
if len(MODELS) == 1:
    MODELS *= 3

for k in range(3):
    print("==============")

    print("PREDICTION FILE " + str(k))
    start_file = time.time()

    ### Name of features to load if available  
    name_features = "features_" + str(k) + "_kmin_" + str(
        kmer_min_size) + "_kmax_" + str(kmer_max_size)
    if with_misplacement:
        name_features += "_mis_" + str(number_misplacements)

    if not (scaling_features):
        name_features += '_unscaled'
    train_name_features = name_features + "_Xtrain.npy"
    test_name_features = name_features + "_Xtest.npy"

    if use_fast_kmer_process:
        if SUM_KERNEL_SPECTRUM_PARAMS:

            X_train_list = []
            X_test_list = []

            for km_size, nb_mismatch in SUM_KERNEL_SPECTRUM_PARAMS:
                X_train, y_train, X_test = read_dataset_train_test_fast_kmer_process(
                    k, km_size,
                    nb_mismatch,
                    scaling_features=scaling_features,
                    exhaustive_spectrum=exhaustive_spectrum,
                    use_sparse_matrix=use_sparse_matrix)

                X_train_list.append(X_train)
                X_test_list.append(X_test)

            X_train = X_train_list
            X_test = X_test_list

        else:
            assert kmer_min_size == kmer_max_size
            X_train, y_train, X_test = read_dataset_train_test_fast_kmer_process(
                k, kmer_min_size,
                number_misplacements, scaling_features,
                use_sparse_kmer_process)

    elif os.path.isfile(FEATURE_FILE_PREFIX + train_name_features):
        print("LOADING FEATURES")
        X_train, y_train, X_test = load_data_for_submission(train_name_features,
                                                            test_name_features)

    checkpoint_1 = time.time()
    print("TIME FOR EXTRACTION " + str(k) + " : " + str(
        int(checkpoint_1 - start_file)) + " seconds")

    ### Choice of Kernel
    kernel_selected = MODELS[k]
    print(
        f"TRAINING MODEL: {kernel_selected.__class__.__name__} "
        f"{kernel_selected.__dict__}")

    ### Kernel fitting
    print("FITTING")
    kernel_selected.fit(X_train, y_train)
    checkpoint_2 = time.time()
    print("TIME FOR FITTING " + str(k) + " : " + str(
        int(checkpoint_2 - checkpoint_1)) + " seconds")

    ### Prediction on train set
    y_train_pred = kernel_selected.predict(X_train)
    train_accuracy = accuracy_score(y_train_pred, y_train)
    print(f"\tTrain {train_accuracy * 100:.1f}%")

    ### Prediction on test set
    print("PREDICTING")
    test_prediction[k] = kernel_selected.predict(X_test)
    checkpoint_3 = time.time()
    print("TIME FOR PREDICTING " + str(k) + " : " + str(
        int(checkpoint_3 - checkpoint_2)) + " seconds")

    runtime_file = time.time() - start_file
    print("TOTAL TIME FOR FILE " + str(k) + " : " + str(
        int(runtime_file)) + " seconds")

# %% Create submission in right format

submission_name = "sum_6_1_9_1_lin_weighted_log.csv"

id_test = [i for i in range(3000)]
prediction_test = list(test_prediction[0]) + list(test_prediction[1]) + list(
    test_prediction[2])

submission_df = pd.DataFrame(data={'ID': id_test, 'Bound': prediction_test},
                             columns=['ID', 'Bound'])

submission_df.to_csv("submissions/" + submission_name, index=False)
