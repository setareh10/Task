# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:40:30 2023

@author: setareh
"""
import pandas as pd
from sklearn.decomposition import PCA

from functions import scale_data, data_preprocessing, population_identification
from sklearn.cluster import KMeans
from datasets import dataset_A_new, dataset_D_new
from Question1B import classifier

dataset_A_classified, clonal_count, mixed_count = classifier(
    dataset_A_new, sparsity_degree=3)

dataset_A_mixed = dataset_A_classified[dataset_A_classified['mixed_infection'] == "Mixed"]


pca_a, a_scaled = data_preprocessing(dataset_A_mixed, PCA, n_components=25)
pca_d, d_scaled = data_preprocessing(dataset_D_new, pca_a, n_components=None)


categories = population_identification (a_scaled, d_scaled, n_clusters=2)



for i in range(categories.shape[0]):

    print('sample ' + str(i) + ' belongs to population ' + str(categories[i]))

