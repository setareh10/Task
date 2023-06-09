# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:38:30 2023

@author: setareh
"""

from sklearn.decomposition import PCA
from datasets import dataset_C_new, dataset_A_new
from Question1A import populations_identification_kmeans
from functions import data_preprocessing, population_identification


if __name__ == "__main__":
    
    ## Prepare the datasets
    pca_a, a_scaled = data_preprocessing(dataset_A_new, PCA, n_components=25)
    pca_c, c_scaled = data_preprocessing(dataset_C_new, pca_a, n_components=None)
    
    ## Determine the optimum number of clusters
    n_clusters = populations_identification_kmeans(a_scaled)
    
    ## Perform k-means clustering
    categories = population_identification (a_scaled, c_scaled, n_clusters)
    
    for i in range(categories.shape[0]):
    
        print('sample ' + str(i) + ' belongs to population ' + str(categories[i]))
