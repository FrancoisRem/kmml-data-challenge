import random
from copy import deepcopy

import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV

from kmer_processor import *
from models import *
from utils import *

# Path prefix constants
DATA_FILE_PREFIX = "../data/"
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
test_size = 0.20
scaling_features = False

use_sparse_matrix = True
exhaustive_spectrum = True
do_cross_val_grid_search = False
cross_val_kfold_k = 5

# If not empty, --> use sum kernel: provide list of kernel as kernel with the
# same size as SUM_KERNEL_PARAMS.
SUM_KERNEL_SPECTRUM_PARAMS = False #[(7, 1), (9, 1)]
SUM_KERNEL_KERNELS = [(GAUSSIAN_KERNEL, 0.3), (GAUSSIAN_KERNEL, 0.7)]

# Models to benchmark Train/Test evaluation.
list_alpha = [1e-7, 1e-6, 5*1e-6, 1e-5, 5*1e-5, 7.5*1e-5, 1e-4, 2.5*1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2, 5*1e-2]

#list_alpha = [1,200, 4000]

#list_alpha= [0.00001, 0.0005, 0.001]

MODELS = [KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=x) for x in list_alpha]
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-7),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=5*1e-7),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-6),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=5*1e-6),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-5),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=5*1e-5),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-4),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=5*1e-4),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-3),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=5*1e-3),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-2),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=5*1e-2),
#     KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-1),
# ]

# Model and parameters to benchmark using cross-validation grid-search.
CV_MODEL = KernelSVMClassifier()
CV_TUNED_PARAMS = [
    {'kernel': [GAUSSIAN_KERNEL], 'alpha': list_alpha}]

# %% RUN FULL PIPELINE

train_dict = {}
test_dict = {}
number_experiments = 10


for exp in range(number_experiments):
    train_dict[exp] = {}
    test_dict[exp] = {}
    for k in range(1):
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
                n_jobs=5, cv=cross_val_kfold_k
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
                f"\t Train {train_accuracy * 100:.1f}% "
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
                
                train_dict[exp][model.__dict__['alpha_']] = train_accuracy
                test_dict[exp][model.__dict__['alpha_']] = test_accuracy
    
                print(
                    f"\tTrain {train_accuracy * 100:.1f}% "
                    f"| Test:{test_accuracy * 100:.1f}%")


#%% 
import matplotlib.pyplot as plt

train_accuracy_median = [np.median([train_dict[exp][alpha] for exp in range(number_experiments)]) for alpha in list_alpha]
train_accuracy_std = [np.std([train_dict[exp][alpha] for exp in range(number_experiments)]) for alpha in list_alpha]
test_accuracy_median = [np.median([test_dict[exp][alpha] for exp in range(number_experiments)]) for alpha in list_alpha]
test_accuracy_std = [np.std([test_dict[exp][alpha] for exp in range(number_experiments)]) for alpha in list_alpha]


plt.figure(figsize=(20,8), dpi=200)
fig, ax1 = plt.subplots()
ax1.set_xscale('log')
ax1.set_xlabel(r'$\alpha$', fontsize = 18)
plt.xticks(fontsize=14)
ax1.set_xlim(left = min(list_alpha), right = max(list_alpha))
ax1.set_ylabel("Train accuracy", color='b', fontsize = 20, rotation=90)
#ax1.set_ylim(top = 6.5, bottom = 4.5)
plt.yticks([0.50,0.60,0.70,0.80,0.90,1.0],fontsize=13)
#ax1.plot(list_alpha, train_accuracy_median, color='b',alpha=0.8)
ax1.errorbar(list_alpha, train_accuracy_median, yerr=train_accuracy_std, color='b',ecolor='b', capsize=3)

ax2 = ax1.twinx()  
ax2.set_ylabel("Test accuracy", color='r', fontsize = 20, rotation=90)  # we already handled the x-label with ax1
ax2.set_ylim(bottom=0.45,top = 0.70)
plt.yticks([0.45,0.50,0.55,0.60,0.65, 0.70],fontsize=13)
#ax2.plot(list_alpha, test_accuracy_median, color='r',alpha=0.8)
ax2.errorbar(list_alpha, test_accuracy_median, yerr=test_accuracy_std, color='r',ecolor='r', capsize=3)
plt.savefig("behaviour_plot_v3.png", dpi=400, bbox_inches='tight')

plt.show()

#test_accuracy_std_2= [0.1]*len(test_accuracy_median)

#plt.errorbar(list_alpha, test_accuracy_median, yerr=test_accuracy_std_2, color='r',ecolor='r')
#plt.show()
