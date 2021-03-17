#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:13:56 2021

@author: theophanegregoir
"""

#%% IMPORTS

import pandas as pd
import numpy as np

#%% MAJORITY VOTING

SUBMISSION_FOLDER = "submissions/"

def submissions_go_to_the_ballots(list_submissions_name, majority_submission_name):
    
    ### Load all DataFrames of submission
    list_submission_df = []
    for submission_name in list_submissions_name:
        list_submission_df.append(pd.read_csv(SUBMISSION_FOLDER+submission_name))
    
    ### Voting process
    vote_df = list_submission_df[0].copy()
    vote_df['Sum'] = 0 
    
    for submission_df in list_submission_df :
        vote_df['Sum'] += submission_df['Bound']
    
    ### Prepare submission
    final_df = list_submission_df[0].copy()
    final_df['Bound'] = (vote_df['Sum'] > (len(list_submissions_name) / 2.0)).astype(int)
    
    final_df.to_csv(SUBMISSION_FOLDER+majority_submission_name, index=False)
    
    
#%% Run the majority voting

list_submissions_name = ["657_submission_6kmer_2mis_rbf_svm.csv",
                         "658_submission_7kmer_2mis_rbf_svm.csv",
                         "651_submission_5kmer_1mis_rbf_svm.csv",
                         "647_submission_100mat_kmer_4_6_misplacement1_SVM.csv"
                         ]

majority_submission_name = "piche.csv"

submissions_go_to_the_ballots(list_submissions_name, majority_submission_name)    




