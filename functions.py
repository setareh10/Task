# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:08:29 2023

@author: setareh
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def scale_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def replace_missing_values(df):

    df.replace("-", np.nan, inplace=True)
    df_new = df.apply(lambda row: row.fillna(row.mode()[0]), axis=1)

    return df_new


def data_preprocessing(df, PCA, n_components):
    df_dummies = pd.get_dummies(df)
    if n_components is None:
        pca = PCA
    else:
        pca = PCA(n_components)
    df_reduced = pca.fit_transform(df_dummies)
    # print(f'Total explained variance: {(pca.explained_variance_ratio_).sum()}')
    df_scaled = scale_data(df_reduced)
    
    return (pca, df_scaled)



def label_mixed_clonal(df, col, sparsity_degree):
    for index, row in df.iterrows():
        if df.at[index, col] > sparsity_degree:
            df.at[index, "mixed_infection"] = "Mixed"

    return df


def count_heterozygous(df):

    for index, row in df.iterrows():
        h_count = 0
        population = set()
        for value in row:
            if '/' in str(value):
                population.add(value)
                h_count = h_count + 1

        df.at[index, "heterozygous_count"] = h_count
        df.at[index, "heterozygous_unique"] = len(population)

    return df


def population_identification (train, test, n_clusters):

    model = KMeans(n_clusters=n_clusters)
    model.fit(train)
    
    categories = model.predict(test)
    return categories