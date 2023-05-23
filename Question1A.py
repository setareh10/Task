# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:02:47 2023

@author: setareh
"""
from sklearn.cluster import KMeans
from datasets import dataset_A_new
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from functions import data_preprocessing
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage


def populations_identification_hierarchical(df):
    """
    Perform hierarchical clstering to identify different popolations/clusters.

    Parameters
    ----------
    df : DataFrame.

    Returns
    -------
    None.
    """
    
    Z = linkage(df, method='ward')
    fig = plt.figure(figsize=(25, 10))
    plt.title('Hierarchical clustering of Sudorphidius parasites', fontsize=18)
    dn = dendrogram(Z)
    
    return None
    

def populations_identification_kmeans(df):
    """
    Perform k-means clstering to identify the number of different 
    popolations/clusters determined by elbow method.


    Parameters
    ----------
    df : DataFrame.

    Returns
    -------
    n_clusters: int.

    """
 
    x0 = df.copy()
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 20))
    visualizer.fit(x0)
    n_clusters = visualizer.elbow_value_
    
    return int(n_clusters)


if __name__ == "__main__":
    
    ## Prepare the data
    pca, a_scaled = data_preprocessing(dataset_A_new, PCA, n_components=25)
    
    populations_identification_hierarchical(a_scaled)
    
    n_clusters = populations_identification_kmeans(a_scaled)
    
    
    if n_clusters > 1:
        print(f"Different populations (~{n_clusters}) have been identified!")
    else:
        print("One single population has been identified!")
