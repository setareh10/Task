# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:02:47 2023

@author: setareh
"""
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from datasets import dataset_A_new
from functions import scale_data, data_preprocessing


    
def populations_identification_hierarchical(df, n_components):
    

    pca, a_scaled = data_preprocessing(df, PCA, n_components)
    
    Z = linkage(a_scaled, method='ward')
    fig = plt.figure(figsize=(25, 10))
    plt.title('Hierarchical clustering of Sudorphidius parasites', fontsize=18)
    dn = dendrogram(Z)
    return 
    

def populations_identification_kmeans(df, n_components):
 
    pca, a_scaled = data_preprocessing(df, PCA, n_components)

    x0 = a_scaled.copy()
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 20))
    visualizer.fit(x0)
    n_clusters = visualizer.elbow_value_
    
    return int(n_clusters)



populations_identification_hierarchical(dataset_A_new, n_components=25)

n_clusters = populations_identification_kmeans(dataset_A_new, n_components=25)


if n_clusters > 1:
    print(f"Different populations (~{n_clusters}) have been identified!")
else:
    print("One single population has been identified!")
