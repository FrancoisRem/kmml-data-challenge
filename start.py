"""
Created on Sun Feb 28 13:58:29 2021

@author: theophanegregoir
"""

# %% Imports

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

    # When processing feature embeddings with mismatches, account for patterns
    # which do not appear in any string (but are m or less mismatches away from 
    # an existing one) or just consider appearing patterns
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


# %% SELECTING PARAMETERS

# Parameters in case a signe kernel is used
kmer_size = 8
number_misplacements = 2

# Scale the feature embeddings before entering the kernel or not
scaling_features = False

# Use Scipy Sparse matrix to optimize storage or not
use_sparse_matrix = True

# When processing feature embeddings with mismatches, account for patterns
# which do not appear in any string (but are m or less mismatches away from 
# an existing one) or just consider appearing patterns
exhaustive_spectrum = True

# If not empty, --> use sum kernel: provide list of kernel as kernel with the
# same size as SUM_KERNEL_PARAMS.

# (kmer size, number of misplacements)
SUM_KERNEL_SPECTRUM_PARAMS = [(6, 1), (9, 1)]
SUM_KERNEL_KERNELS = [(LINEAR_KERNEL, 1e-2), (LINEAR_KERNEL, 1)]

# Models to be used for submission
MODELS = [
    KernelLogisticClassifier(kernel=SUM_KERNEL_KERNELS, alpha=0.5),
]

# %% FITTING AND PREDICTING

### Initializing final variable containing the 3000 predictions
test_prediction = {}

# Handle single-model case.
if len(MODELS) == 1:
    MODELS *= 3

for k in range(3):
    print("==============")

    print("PREDICTION FILE " + str(k))
    start_file = time.time()

    ### Extraction of features
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
        X_train, y_train, X_test = read_dataset_train_test_fast_kmer_process(k,
                                                                             kmer_size,
                                                                             number_misplacements,
                                                                             scaling_features=scaling_features,
                                                                             exhaustive_spectrum=exhaustive_spectrum,
                                                                             use_sparse_matrix=use_sparse_matrix)

    checkpoint_1 = time.time()
    print("TIME FOR EXTRACTION " + str(k) + " : " + str(
        int(checkpoint_1 - start_file)) + " seconds")

    ### Choice of Kernel
    kernel_selected = MODELS[k]
    print(
        f"TRAINING MODEL: {kernel_selected.__class__.__name__} "
        f"{kernel_selected.__dict__}")

    ### Kernel fitting
    kernel_selected.fit(X_train, y_train)
    checkpoint_2 = time.time()
    print("TIME FOR FITTING " + str(k) + " : " + str(
        int(checkpoint_2 - checkpoint_1)) + " seconds")

    ### Prediction on train set
    y_train_pred = kernel_selected.predict(X_train)
    train_accuracy = accuracy_score(y_train_pred, y_train)
    print(f"\tTrain {train_accuracy * 100:.1f}%")

    ### Prediction on test set
    test_prediction[k] = kernel_selected.predict(X_test)

    runtime_file = time.time() - start_file
    print("TOTAL TIME FOR FILE " + str(k) + " : " + str(
        int(runtime_file)) + " seconds")

# %% CREATING SUBMISSION

### Submission name to produce
submission_name = "Professor_submission.csv"

id_test = [i for i in range(3000)]
prediction_test = list(test_prediction[0]) + list(test_prediction[1]) + list(
    test_prediction[2])

submission_df = pd.DataFrame(data={'ID': id_test, 'Bound': prediction_test},
                             columns=['ID', 'Bound'])

submission_df.to_csv("submissions/" + submission_name, index=False)
