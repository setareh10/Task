# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:02:47 2023

@author: setar
"""
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from datasets import dataset_A_new
from functions import scale_data


a_dummies = pd.get_dummies(dataset_A_new)
pca = PCA(25)
a_prime = pca.fit_transform(a_dummies)
print(f'Total explained variance: {(pca.explained_variance_ratio_).sum()}')

a_scaled = scale_data(a_prime)

Z = linkage(a_scaled, method='ward')
fig = plt.figure(figsize=(25, 10))
plt.title('Hierarchical clustering of Sudorphidius parasites', fontsize=18)
dn = dendrogram(Z)


x0 = a_scaled.copy()
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 20))
visualizer.fit(x0)
n_clusters = visualizer.elbow_value_

if n_clusters > 1:
    print("Different populations have been identified!")
else:
    print("One single population has been identified!")
