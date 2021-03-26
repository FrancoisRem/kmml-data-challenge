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

list_submissions_name = ["670_combination_7kmer_1mis_3_SVM_rbf.csv",
                         "668_submission_7kmer_2mis_rbf_svm_tailored.csv",
                         "663_sum_10_2_7_1_rbf_SVM.csv",
                         "669_6_1_9_1_lin_Log.csv",
                         "679_sum_6_1_9_1_lin_weighted_log.csv"]

majority_submission_name = "majority_voting_top_5.csv"

submissions_go_to_the_ballots(list_submissions_name, majority_submission_name)    




