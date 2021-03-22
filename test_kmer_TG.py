from feature_extractor import *
from kmer_processor import *
from models import *

#%% Load data

DATA_FILE_PREFIX= "data/"
FEATURE_FILE_PREFIX= "features/"
TRAINING_FILE_PREFIX="Xtr"
LABEL_FILE_PREFIX="Ytr"
TEST_FILE_PREFIX="Xte"

def read_dataset_train_test(k, use_mat_features=True, use_kmers=True, kmer_min_size=3, kmer_max_size=4, with_misplacement=True, number_misplacements=1, dict_original_pattern_to_misplaced=None, scaled=False):
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
  
  if scaled :
      print("SCALING")
      X_train, X_test = standardize_train_test(X_train, X_test)
  
    
  return X_train, y_train, X_test, dict_original_pattern_to_misplaced


def read_train_dataset(k):
    Xtr_df = pd.read_csv(
        DATA_FILE_PREFIX + TRAINING_FILE_PREFIX + str(k) + ".csv")
    Ytr_df = pd.read_csv(DATA_FILE_PREFIX + LABEL_FILE_PREFIX + str(k) + ".csv")
    return pd.merge(left=Xtr_df, right=Ytr_df, on='Id')

#%%
dict_original_pattern_to_misplaced = None

kmer_size = 4
with_misplacement = True
mispl = 2


for k in range(3):
    ### Old TG way
    kernel_old = LinearKernelBinaryClassifier(kernel="lin")
    
    X_train, y_train, X_test, dict_original_pattern_to_misplaced = read_dataset_train_test(k,
                                                       use_mat_features=False,
                                                       use_kmers=True,
                                                       kmer_min_size=kmer_size,
                                                       kmer_max_size=kmer_size,
                                                       with_misplacement=with_misplacement,
                                                       number_misplacements=mispl,
                                                       dict_original_pattern_to_misplaced=dict_original_pattern_to_misplaced)
    
    Gram_matrix_old = kernel_old._gram_matrix(X_train, X_train)
    
    ### New FR way
    kernel_new = LinearKernelBinaryClassifier(kernel="lin")
    
    ### compare Gram matrix
    df = read_train_dataset(k)
    processor = KMerProcessor(df['seq'])
    spectrums = processor.compute_kmer_mismatch(kmer_size,mispl)
    X_train_bis = compute_spectrums_matrix(processor.kmers_support,
                                                spectrums)
    
    Gram_matrix_new = kernel_new._gram_matrix(X_train_bis, X_train_bis)
    
    diff_Gram = Gram_matrix_new - Gram_matrix_old
    diff_X = X_train_bis - X_train
    
    print(diff_Gram[:5,:5])





