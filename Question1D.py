# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:40:30 2023

@author: setareh
"""
from Question1B import classifier
from sklearn.decomposition import PCA
from datasets import dataset_A_new, dataset_D_new
from functions import data_preprocessing, population_identification, count_heterozygous


if __name__ == "__main__":
    
    df_A, _ = count_heterozygous(dataset_A_new)

    ## Classify the dataset_A as being "Clonal" or "Mixed" infections
    dataset_A_classified, clonal_count, mixed_count = classifier(
        df_A, sparsity_degree=3)
    
    ## Extract the mixed infections samples to build a new dataframe
    dataset_A_mixed = dataset_A_classified[dataset_A_classified['mixed_infection'] == "Mixed"]
    
    ## Prepare the datasets
    pca_a, a_scaled = data_preprocessing(dataset_A_mixed, PCA, n_components=25)
    pca_d, d_scaled = data_preprocessing(dataset_D_new, pca_a, n_components=None)
    
    ## Perfomrs the k-means clustering for two clusters/populations
    categories = population_identification(a_scaled, d_scaled, n_clusters=2)
    
    
    
    for i in range(categories.shape[0]):
    
        print(f'sample {i} belongs to population {categories[i]}')
    
