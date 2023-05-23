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
    """
    Standardise the data
    Parameters
    ----------
    df : DataFrame.
    
    Returns
    -------
    df_scaled : DataFrame.

    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    return df_scaled


def replace_missing_values(df):
    """
    Replace missing values of each row "-" with 
    the most frequent DNA marker of the row.

    Parameters
    ----------
    df : DataFrame.

    Returns
    -------
    df_new : DataFrame.

    """

    df.replace("-", np.nan, inplace=True)
    df_new = df.apply(lambda row: row.fillna(row.mode()[0]), axis=1)

    return df_new


def data_preprocessing(df, PCA, n_components):
    """
    Converts the categorical data into numerical,
    reduce the dimensions of the data using PCA,
    and standardise the data. 

    Parameters
    ----------
    df : DataFrame.
    PCA : PCA object.
    n_components : int, or float. 
    Number of components to keep.  If 0 < n_components < 1 , select the number 
    of components such that the amount of variance that needs to be explained
    is greater than the percentage specified by n_components 

    Returns
    -------
    pca : PCA object.
    df_scaled : DataFrame.

    """
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
    """
    Label each row as being "mixed" or "clonal" based on the number of unique 
    heterozygous genotypes and the maximum number of heterozygous genotypes
    allowed in clonal infections specified by sparsity_degree.

    Parameters
    ----------
    df : DataFrame.
    col : string.
    sparsity_degree : int.

    Returns
    -------
    df : DataFrame.

    """
    for index, row in df.iterrows():
        if df.at[index, col] > sparsity_degree:
            df.at[index, "mixed_infection"] = "Mixed"

    return df


def count_heterozygous(df):
    """
    Count the number of (unique) heterozygous genotypes in each row.

    Parameters
    ----------
    df : DataFrame.

    Returns
    -------
    df : DataFrame.

    """

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
    """
    Determines which population/cluster each unseen sample (test) belongs to.

    Parameters
    ----------
    train : DataFrame.
    test : DataFrame.
    n_clusters : int.

    Returns
    -------
    categories : Array of int32.

    """

    model = KMeans(n_clusters=n_clusters)
    model.fit(train)
    
    categories = model.predict(test)
    return categories