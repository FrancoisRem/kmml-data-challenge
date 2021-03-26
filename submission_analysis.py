#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:37:27 2021

@author: theophanegregoir
"""

#%% IMPORTS

import pandas as pd
import numpy as np

#%% Analyse submissions

def analyse_two_submissions(path_sub1, path_sub2):
    
    ### Load submissions
    SUBMISSION_FOLDER = "submissions/"
    sub1_df = pd.read_csv(SUBMISSION_FOLDER+path_sub1)
    sub2_df = pd.read_csv(SUBMISSION_FOLDER+path_sub2)
    
    print("==============")
    print("Comparison : " + path_sub1 + " VS " + path_sub2)
    
    ### Matrix of difference between submissions
    confusion_df = sub1_df.copy()
    confusion_df['Bound_1'] = sub1_df['Bound']
    confusion_df['Bound_2'] = sub2_df['Bound']
    confusion_df = confusion_df.drop(columns=['Bound'])
    confusion_df['Difference'] = (confusion_df['Bound_1'] != confusion_df['Bound_2'])
    
    ### Analysis per dataset given
    for k in range(3):
        df_selected = confusion_df[(confusion_df['ID'] >= k*1000) & (confusion_df['ID'] < (k+1)*1000)]
        number_disagreements = len(df_selected[df_selected['Difference'] == True])
        print("DATASET " + str(k) + " DISAGREEMENT RATE : " + str(100.0 * number_disagreements / len(df_selected)) + "%")
    
    return confusion_df

    
#%% Run analysis

path_sub1 = "679_sum_6_1_9_1_lin_weighted_log.csv"
path_sub2 = "majority_voting_top_5.csv"

confusion_df = analyse_two_submissions(path_sub1, path_sub2)
