# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:38:30 2023

@author: setareh
"""
import pandas as pd
from datasets import dataset_C_new, dataset_A_new
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from functions import scale_data, data_preprocessing
from Question1A import populations_identification_kmeans


pca_a, a_scaled = data_preprocessing(dataset_A_new, PCA, n_components=25)
pca_c, c_scaled = data_preprocessing(dataset_C_new, pca_a, n_components=None)

n_clusters = populations_identification_kmeans(dataset_A_new, n_components=25)


def population_identification (train, test, n_clusters):

    model = KMeans(n_clusters=n_clusters)
    model.fit(train)
    
    categories = model.predict(test)
    return categories


categories = population_identification (a_scaled, c_scaled, n_clusters)

for i in range(categories.shape[0]):

    print('sample ' + str(i) + ' belongs to population ' + str(categories[i]))
