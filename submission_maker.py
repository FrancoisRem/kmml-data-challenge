"""
Created on Sun Feb 28 13:58:29 2021

@author: theophanegregoir
"""

#%% Imports

from models import *
from feature_extractor import *

import pandas as pd
import numpy as np


#%% Load data

DATA_FILE_PREFIX= "data/"
TRAINING_FILE_PREFIX="Xtr"
LABEL_FILE_PREFIX="Ytr"
TEST_FILE_PREFIX="Xte"

def read_dataset_train_test(k, use_mat_features=True, use_kmers=True, kmer_max_size=4):
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
      list_kmer_pattern, df_train = add_kmer_features(kmer_max_size, df_train)
      
      ### test
      _, df_test = add_kmer_features(kmer_max_size, df_test)
      
      list_features += list_kmer_pattern

  ### Create X_train and y_train
  y_train = df_train['Bound'].to_numpy()
  
  ### Center data and to numpy
  scaled_train = (df_train[list_features] - df_train[list_features].mean())/df_train[list_features].std()
  X_train = scaled_train.to_numpy()
  
  scaled_test = (df_test[list_features] - df_test[list_features].mean())/df_test[list_features].std()
  X_test = scaled_test.to_numpy()
  
  return X_train, y_train, X_test

#%% Actual Fitting and prediction

### 3 different datasets
test_prediction = {}
for k in range(3):
    print("==============")
    print("PREDICTION FILE " + str(k))
    ### Choice of Kernel
    kernel_selected = KernelSVMClassifier()
    
    ### Dataset loader with feature choice
    X_train, y_train, X_test = read_dataset_train_test(k, use_mat_features=True)# use_kmers=True, kmer_max_size=4)
    
    ### Kernel fitting
    kernel_selected.fit(X_train, y_train)
    
    ### Prediction on test set
    test_prediction[k] = kernel_selected.predict(X_test)

#%% Create submission in right format

submission_name = "submission_withoutkmer_SVM.csv"

id_test = [i for i in range(3000)]
prediction_test = list(test_prediction[0]) + list(test_prediction[1]) + list(test_prediction[2])

submission_df = pd.DataFrame(data={'ID': id_test, 'Bound' : prediction_test}, columns=['ID', 'Bound'])

submission_df.to_csv("submissions/"+ submission_name, index=False)