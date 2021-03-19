import os
import random
from copy import deepcopy

from sklearn.model_selection import GridSearchCV

from feature_extractor import *
from models import *

# Global initialization
SEED = 2021

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


def compute_features(df,
                     use_mat_features=False,
                     use_kmers=True,
                     kmer_min_size=4, kmer_max_size=5, with_misplacement=True,
                     number_misplacements=1,
                     dict_original_pattern_to_misplaced=None):
    list_features = []
    # Features created by professors
    if use_mat_features:
        Xtr_mat100_df = pd.read_csv(
            DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + "_mat100.csv",
            sep=' ', header=None)
        df[['feature_given_' + str(j) for j in range(1, 101)]] = Xtr_mat100_df
        list_features += ['feature_given_' + str(j) for j in range(1, 101)]

    # Features of kmers frequency
    if use_kmers:
        list_kmer_pattern, df, dict_original_pattern_to_misplaced = add_kmer_features(
            kmer_min_size, kmer_max_size,
            with_misplacement,
            number_misplacements, df, dict_original_pattern_to_misplaced)
        list_features += list_kmer_pattern

    return df, list_features, dict_original_pattern_to_misplaced


def split_train_test(df, list_features, test_size=0.20):
    # Split train test
    df_test = df.sample(frac=test_size, random_state=SEED)
    df_train = df.drop(df_test.index)

    # Create train and test label vectors
    y_train = df_train['Bound'].to_numpy()
    y_test = df_test['Bound'].to_numpy()

    # Create train and test feature vectors
    X_train = df_train[list_features].to_numpy()
    X_test = df_test[list_features].to_numpy()

    return X_train, X_test, y_train, y_test


def load_data(train_name_features, test_size=0.20):
    ### Load full feature matrix
    total_feature_array = np.load(FEATURE_FILE_PREFIX + train_name_features)

    ### Load full label matrix
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    total_label_array = Ytr_df['Bound'].to_numpy()

    ### Split Train and Validation
    test_idx = random.sample(range(0, 2000),
                             int(total_label_array.shape[0] * test_size))
    train_idx = [i for i in range(0, 2000) if not (i in test_idx)]

    X_train = total_feature_array[train_idx, :]
    y_train = total_label_array[train_idx]

    X_test = total_feature_array[test_idx, :]
    y_test = total_label_array[test_idx]

    return X_train, X_test, y_train, y_test


# %% SELECT FEATURES

use_mat_features = False
use_kmers = True
kmer_min_size = 7
kmer_max_size = 7
with_misplacement = True
number_misplacements = 1
test_size = 0.20
do_cross_val_grid_search = False
cross_val_kfold_k = 5

# Models to benchmark Train/Test evaluation.
MODELS = [
    KernelSVMClassifier(kernel='rbf', alpha=2 * 1e-4),
    KernelLogisticClassifier(kernel='rbf', alpha=2 * 1e-4),
]

# Model and parameters to benchmark using cross-validation grid-search.
CV_MODEL = KernelSVMClassifier()
CV_TUNED_PARAMS = [
    {'kernel': [LINEAR_KERNEL], 'alpha': [1e-1, 1e-2, 1e-3]}]

# %% RUN FULL PIPELINE

dict_original_pattern_to_misplaced = None
for k in range(3):

    # Reinitialize models at each iteration
    models = deepcopy(MODELS)
    print(f"------PREDICTION FILE {k}------")

    ### Check if features were already computed and saved
    name_features = "features_" + str(k) + "_kmin_" + str(
        kmer_min_size) + "_kmax_" + str(kmer_max_size)
    if with_misplacement:
        name_features += "_mis_" + str(number_misplacements)
    train_name_features = name_features + "_Xtrain.npy"

    if os.path.isfile(FEATURE_FILE_PREFIX + train_name_features):
        print("Features already computed and saved : loading...")
        X_train, X_test, y_train, y_test = load_data(train_name_features,
                                                     test_size=test_size)

    ### Otherwise compute them
    else:
        print("No previous similar computation : computing...")
        df = read_train_dataset(k)

        df, list_features, dict_original_pattern_to_misplaced = compute_features(
            df, use_mat_features,
            use_kmers, kmer_min_size, kmer_max_size, with_misplacement,
            number_misplacements, dict_original_pattern_to_misplaced)

        X_train, X_test, y_train, y_test = split_train_test(df, list_features)

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
