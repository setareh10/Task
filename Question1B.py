# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:32:18 2023

@author: setareh
"""
import pandas as pd
from datasets import dataset_A_new
from sklearn.decomposition import PCA
from functions import (count_heterozygous, label_mixed_clonal, 
    population_identification, data_preprocessing)
from Question1A import populations_identification_kmeans


def classifier(df, sparsity_degree):
    """
    Label each sample as "Clonal" or "Mixed"

    Parameters
    ----------
    df : DataFrame.
    sparsity_degree : int.

    Returns
    -------
    df_classified : DataFrame.
    clonal_count : int.
    mixed_count : int.

    """

    df["mixed_infection"] = "Clonal"

    df_classified = label_mixed_clonal(
        df, "heterozygous_unique", sparsity_degree)

    infection_nb = df_classified["mixed_infection"].value_counts()
    clonal_count = infection_nb[0]
    mixed_count = infection_nb[1]

    return df_classified, clonal_count, mixed_count


## Create a new dataframe with new columns, counting heterozygous genotypes
df_A, heterozygous_genotypes = count_heterozygous(dataset_A_new)

## Perfomr simple classification/labelling
dataset_A_classified, clonal_count, mixed_count = classifier(
    df_A, sparsity_degree=3)


if clonal_count > mixed_count:
    print(
        f'Infections are primarily clonal, with {clonal_count} samples out of {clonal_count+mixed_count}!')
else:
    print(
        'Infections are primarily mixed, with {mixed_count} samples out of {clonal_count+mixed_count}!')


## To find out if diffenet populations have different prevalence of mixed infections

pca_a, a_scaled = data_preprocessing(dataset_A_new, PCA, n_components=25)

n_clusters = populations_identification_kmeans(a_scaled)

categories = population_identification(a_scaled, a_scaled, n_clusters)

population_df = pd.DataFrame(list(zip(heterozygous_genotypes, list(categories))),
               columns =['heterozygous genotypes', 'populations'])

population_df = population_df.sort_values('populations')
