# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:07:57 2023

@author: setar
"""

import pathlib
import pandas as pd
from functions import replace_missing_values


# change it to the path to the data directory
DATA_PATH = "C:\\Users\\setar\\Desktop\\ML task\\florinov-dc"  
if not DATA_PATH:
    raise NotImplementedError("DATA_PATH must be defined!")

path = pathlib.Path(DATA_PATH)

dataset_A = pd.read_csv(path.joinpath("dataset-A.tsv"), sep='\t')
dataset_C = pd.read_csv(path.joinpath("dataset-C.tsv"), sep='\t')
dataset_D = pd.read_csv(path.joinpath("dataset-D.tsv"), sep='\t')
dataset_E = pd.read_csv(path.joinpath("dataset-E.tsv"), sep='\t')


dataset_A = dataset_A.drop(['Unnamed: 0'], axis=1)
dataset_C = dataset_C.drop(['Unnamed: 0'], axis=1)
dataset_D = dataset_D.drop(['Unnamed: 0'], axis=1)
dataset_E = dataset_E.drop(['Unnamed: 0'], axis=1)

dataset_A_new = replace_missing_values(dataset_A)
dataset_C_new = replace_missing_values(dataset_C)
dataset_D_new = replace_missing_values(dataset_D)
dataset_E_new = replace_missing_values(dataset_E)
