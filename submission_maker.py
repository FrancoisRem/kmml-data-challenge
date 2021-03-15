"""
Created on Sun Feb 28 13:58:29 2021

@author: theophanegregoir
"""

#%% Imports

import pandas as pd
import time

from feature_extractor import *
from models import *


#%% Load data

DATA_FILE_PREFIX= "data/"
TRAINING_FILE_PREFIX="Xtr"
LABEL_FILE_PREFIX="Ytr"
TEST_FILE_PREFIX="Xte"

def read_dataset_train_test(k, use_mat_features=True, use_kmers=True, kmer_min_size=3, kmer_max_size=4, with_misplacement=True, number_misplacements=1, dict_original_pattern_to_misplaced=None):
  df_test = pd.read_csv(DATA_FILE_PREFIX+TEST_FILE_PREFIX + str(k) + ".csv")
  
  Xtr_df = pd.read_csv(DATA_FILE_PREFIX+TRAINING_FILE_PREFIX + str(k) + ".csv")
  Ytr_df = pd.read_csv(DATA_FILE_PREFIX+LABEL_FILE_PREFIX + str(k) + ".csv")
  df_train = pd.merge(left=Xtr_df, right=Ytr_df, on='Id')
  
  list_features = []
  
  ### Features created by professors
  if use_mat_features :
      ### train
      Xtr_mat100_df = pd.read_csv(DATA_FILE_PREFIX+TRAINING_FILE_PREFIX + str(k) + "_mat100.csv", sep=' ', header=None)
      df_train[['feature_given_'+str(j) for j in range(1,101)]] = Xtr_mat100_df
      
      ### test
      Xte_mat100_df = pd.read_csv(DATA_FILE_PREFIX+TEST_FILE_PREFIX + str(k) + "_mat100.csv", sep=' ', header=None)
      df_test[['feature_given_'+str(j) for j in range(1,101)]] = Xte_mat100_df
      
      list_features+=['feature_given_'+str(j) for j in range(1,101)]
  
  ### Features of kmers frequency
  if use_kmers :
      ### train
      list_kmer_pattern, df_train, dict_original_pattern_to_misplaced = add_kmer_features(kmer_min_size, kmer_max_size, with_misplacement, number_misplacements, df_train, dict_original_pattern_to_misplaced)
      
      ### test
      _, df_test, dict_original_pattern_to_misplaced = add_kmer_features(kmer_min_size, kmer_max_size, with_misplacement, number_misplacements,  df_test, dict_original_pattern_to_misplaced)
      
      list_features += list_kmer_pattern

  ### Create X_train and y_train
  y_train = df_train['Bound'].to_numpy()
  
  ### Center data and to numpy
  X_train = df_train[list_features].to_numpy()
  X_test = df_test[list_features].to_numpy()
  X_train, X_test = standardize_train_test(X_train, X_test)
  
  #scaled_test = (df_test[list_features] - df_test[list_features].mean())/df_test[list_features].std()
  #X_test = scaled_total.to_numpy()
  #X_train = scaled_total.to_numpy()

  return X_train, y_train, X_test, dict_original_pattern_to_misplaced

#X_tr, y_tr, X_te = read_dataset_train_test(k, use_mat_features=True, use_kmers=True, kmer_min_size=3, kmer_max_size=6, with_misplacement=True, number_misplacements=1)

#%% Actual Fitting and prediction

### 3 different datasets
test_prediction = {}
dict_original_pattern_to_misplaced = None

for k in range(3):
    print("==============")
    print("PREDICTION FILE " + str(k))
    start_file = time.time()
    ### Dataset loader with feature choice
    print("EXTRACT FEATURES")
    X_train, y_train, X_test, dict_original_pattern_to_misplaced = read_dataset_train_test(k,
                                                       use_mat_features=False,
                                                       use_kmers=True,
                                                       kmer_min_size=7,
                                                       kmer_max_size=7,
                                                       with_misplacement=True,
                                                       number_misplacements=3,
                                                       dict_original_pattern_to_misplaced=dict_original_pattern_to_misplaced)

    ### Choice of Kernel
    kernel_selected = KernelSVMClassifier(kernel=GAUSSIAN_KERNEL, alpha=1e-5)
    
    ### Kernel fitting
    print("FITTING")
    kernel_selected.fit(X_train, y_train)
    
    ### Prediction on test set
    print("PREDICTING")
    test_prediction[k] = kernel_selected.predict(X_test)
    
    runtime_file = time.time() - start_file
    print("TIME FOR FILE " + str(k) + " : " + str(int(runtime_file)) + " seconds")

#%% Create submission in right format

submission_name = "submission_7kmer_0mis_rbf_svm.csv"

id_test = [i for i in range(3000)]
prediction_test = list(test_prediction[0]) + list(test_prediction[1]) + list(test_prediction[2])

submission_df = pd.DataFrame(data={'ID': id_test, 'Bound' : prediction_test}, columns=['ID', 'Bound'])

submission_df.to_csv("submissions/"+ submission_name, index=False)