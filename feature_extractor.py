#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 14:10:39 2021

@author: theophanegregoir
"""

#%% Imports
import pandas as pd
import numpy as np
import itertools

#%% K-mers

def occurrences(string, sub):
    '''
    Returns
    -------
    count : number of occurences of sub in string with overlap
    '''
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count

def create_kmer_freq(kmer_patterns, input_string):
    '''
    Parameters
    ----------
    kmer_patterns : dict
        Patterns gathered by length in dict
    input_string : str

    Returns
    -------
    freq_vector : np.array
        Vector of frequencies of kmer patterns in the input
    '''
    freq_vector = []
    for kmer_size in kmer_patterns:
      tot_patterns=len(input_string)-(kmer_size-1)
      for pattern in kmer_patterns[kmer_size]:
        number_occurences = float(occurrences(input_string, pattern))
        freq_vector.append(number_occurences/tot_patterns)  
    return np.array(freq_vector)

def add_kmer_features(kmer_max_size, raw_df):
    '''
    Parameters
    ----------
    kmer_max_size : int
        Maximum length for kmer to be selected
    
    raw_df : pd.DataFrame 
        Raw data with biological sequences as string
    
    Returns
    -------
    list_total_patterns : list
        All feature columns name
    
    df_with_kmers_features : pd.DataFrame
        Data with added features
    '''
    ### Pattern creation to count 
    bases = ["A","T","C","G"]
    kmer_patterns={}
    for i in range(1,kmer_max_size+1):
        kmer_patterns[i] = [''.join(p) for p in itertools.product(bases, repeat=i)]
    
    ### List of all patterns selected
    list_total_patterns = []
    for k in kmer_patterns:
      list_total_patterns+=kmer_patterns[k]
    
    ### Application to the initial data
    df_featured = raw_df.copy()
    df_featured[list_total_patterns] = df_featured.apply(lambda x: create_kmer_freq(kmer_patterns, x['seq']), axis=1, result_type='expand')
    
    return list_total_patterns, df_featured


    
    
        
    




