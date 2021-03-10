from copy import deepcopy

from feature_extractor import *
from models import *

# Global initialization
SEED = 2021

# Path prefix constants
DATA_FILE_PREFIX = "data/"
TRAINING_FILE_PREFIX = "Xtr"
LABEL_FILE_PREFIX = "Ytr"


def read_train_dataset(k):
    Xtr_df = pd.read_csv(
        DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + ".csv")
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    return pd.merge(left=Xtr_df, right=Ytr_df, on='Id')


def compute_features(df,
                     use_mat_features=True,
                     use_kmers=True,
                     kmer_max_size=4):
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
        list_kmer_pattern, df = add_kmer_features(kmer_max_size, df)
        list_features += list_kmer_pattern

    return df, list_features


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


def normalize_train_test(X_train, X_test):
    # Center and scale by X_train statistics
    X_train_mean, X_train_std = X_train.mean(), X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_test


# Models to train and evaluate
MODELS = [
    KernelLogisticClassifier(kernel=LINEAR_KERNEL, alpha=1e-3),
    KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-4),
]
for k in range(3):
    # Reinitialize models at each iteration
    models = deepcopy(MODELS)
    print(f"------PREDICTION FILE {k}------")

    df = read_train_dataset(k)

    df, list_features = compute_features(df, use_mat_features=True)

    X_train, X_test, y_train, y_test = split_train_test(df, list_features)

    X_train, X_test = normalize_train_test(X_train, X_test)

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
