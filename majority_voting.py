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
path_sub1 = "657_submission_6kmer_2mis_rbf_svm.csv"
path_sub2 = "submission_7kmer_3mis_rbf_svm.csv"
path_sub3 = "651_submission_5kmer_1mis_rbf_svm.csv"
sub1_df = pd.read_csv(SUBMISSION_FOLDER+path_sub1)
sub2_df = pd.read_csv(SUBMISSION_FOLDER+path_sub2)
sub3_df = pd.read_csv(SUBMISSION_FOLDER+path_sub3)

confusion_df = sub1_df.copy()
confusion_df['Sum'] = sub1_df['Bound'] + sub2_df['Bound'] + sub3_df['Bound']

majo_sub = sub1_df.copy()
majo_sub['Bound'] = (confusion_df['Sum'] > 1).astype(int)

majo_sub.to_csv("submissions/"+ "majority_voting.csv", index=False)




