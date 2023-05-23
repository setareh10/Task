# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:04:11 2023

@author: setareh
"""
from datasets import dataset_E_new
from functions import data_preprocessing
from sklearn.decomposition import PCA
from Question1A import populations_identification_hierarchical, populations_identification_kmeans
from Question1B import classifier


pca_e, e_scaled = data_preprocessing(dataset_E_new, PCA, n_components=25)


populations_identification_hierarchical(e_scaled)
n_clusters = populations_identification_kmeans(e_scaled)


if n_clusters > 1:
    print(f"Different populations (~{n_clusters}) have been identified!")
else:
    print("One single population has been identified!")



dataset_E_classified, clonal_count, mixed_count = classifier(
    dataset_E_new, sparsity_degree=3)

if clonal_count > mixed_count:
    print(
        f'Infections are primarily clonal, with {clonal_count} samples out of {clonal_count+mixed_count}!')
else:
    print(
        'Infections are primarily mixed, with {mixed_count} samples out of {clonal_count+mixed_count}!')


