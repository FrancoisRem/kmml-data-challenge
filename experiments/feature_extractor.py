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
import time
from tqdm import tqdm
from kmer_processor import hamming_neighborhood

#%% K-mers

def count_misplacements(string1, string2):
    if len(string1)!=len(string2):
        return np.inf
    else :
        return sum(1 for a, b in zip(string1, string2) if a != b)


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
    
    ### List of all patterns selected
    list_total_patterns = []
    for k in kmer_patterns:
      list_total_patterns+=kmer_patterns[k]
    
    dict_count_appearances = {}
    for kmer_size in kmer_patterns:
      for pattern in kmer_patterns[kmer_size]:
          dict_count_appearances[pattern]=0
    
    for kmer_size in kmer_patterns:
        for idx in range(len(input_string)-kmer_size+1):
            dict_count_appearances[input_string[idx:idx+kmer_size]]+=1
    
    freq_values = list(dict_count_appearances.values())
    
    freq_vector = [x for _,x in sorted(zip(list_total_patterns,freq_values))]

    return np.array(freq_vector)

def create_kmer_freq_with_misplacements(kmer_patterns, dict_original_pattern_to_misplaced, input_string):
    '''
    Parameters
    ----------
    kmer_patterns : dict
        patterns gathered by length in dict
    dict_original_pattern_to_misplaced : dict
        orginal pattern key to a list of patterns with fewer misplacements than number_displacements to original pattern
    input_string : str

    Returns
    -------
    freq_vector : np.array
        Vector of frequencies of kmer patterns in the input
    '''
    
    ### List of all patterns selected
    list_total_patterns = []
    for k in kmer_patterns:
      list_total_patterns+=kmer_patterns[k]
    
    ### Calculation without misplacement as classic method
    dict_count_appearances = {}
    for kmer_size in kmer_patterns:
      for pattern in kmer_patterns[kmer_size]:
          dict_count_appearances[pattern]=0
    
    
    for kmer_size in kmer_patterns:
        for idx in range(len(input_string)-kmer_size+1):
            dict_count_appearances[input_string[idx:idx+kmer_size]]+=1
    
    ### Taking misplacement into account
    dict_count_appearances_with_misplacements = {}
    for kmer_size in kmer_patterns:
      for pattern in kmer_patterns[kmer_size]:
          dict_count_appearances_with_misplacements[pattern]=0
          
    for kmer_size in kmer_patterns:
      for pattern_original in kmer_patterns[kmer_size]:
          for pattern_approximate in dict_original_pattern_to_misplaced[pattern_original]:
              dict_count_appearances_with_misplacements[pattern_approximate] += dict_count_appearances[pattern_original]

    freq_values = list(dict_count_appearances_with_misplacements.values())
    
    freq_vector = [x for _,x in sorted(zip(list_total_patterns,freq_values))]

    return np.array(freq_vector)

def add_kmer_features(kmer_min_size, kmer_max_size, with_misplacement, number_misplacements, raw_df, dict_original_pattern_to_misplaced):
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
    for i in range(kmer_min_size,kmer_max_size+1):
        kmer_patterns[i] = [''.join(p) for p in itertools.product(bases, repeat=i)]
    
    ### List of all patterns selected
    list_total_patterns = []
    for k in kmer_patterns:
      list_total_patterns+=kmer_patterns[k]
      
    if with_misplacement :
        if dict_original_pattern_to_misplaced == None :
            
            ### Dico : key : original pattern and value : list of patterns with fewer misplacements than number_displacements
            dict_original_pattern_to_misplaced = {}
            for k in kmer_patterns :              
                print(f"Computing original_pattern_to_misplaced for k={k}:")
                for pattern_original in tqdm(kmer_patterns[k]):
                    dict_original_pattern_to_misplaced[pattern_original] = list(
                        hamming_neighborhood(pattern_original,
                                              number_misplacements))
        #print(dict_original_pattern_to_misplaced["AAA"])
        ### Application to the initial data
        df_featured = raw_df.copy()
        df_featured[list_total_patterns] = df_featured.apply(lambda x: create_kmer_freq_with_misplacements(kmer_patterns, dict_original_pattern_to_misplaced, x['seq']), axis=1, result_type='expand')
    else :
        ### Application to the initial data
        df_featured = raw_df.copy()
        df_featured[list_total_patterns] = df_featured.apply(lambda x: create_kmer_freq(kmer_patterns, x['seq']), axis=1, result_type='expand')
    
    return list_total_patterns, df_featured, dict_original_pattern_to_misplaced


    
    
        
    




