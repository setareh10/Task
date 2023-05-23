# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:32:18 2023

@author: setareh
"""
from datasets import dataset_A_new
from functions import count_heterozygous, label_mixed_clonal


def classifier(df, sparsity_degree):

    count_heterozygous(df)

    df["mixed_infection"] = "Clonal"

    df_classified = label_mixed_clonal(
        df, "heterozygous_unique", sparsity_degree)

    infection_nb = df_classified["mixed_infection"].value_counts()
    clonal_count = infection_nb[0]
    mixed_count = infection_nb[1]

    return df_classified, clonal_count, mixed_count


dataset_A_classified, clonal_count, mixed_count = classifier(
    dataset_A_new, sparsity_degree=3)

if clonal_count > mixed_count:
    print(
        f'Infections are primarily clonal, with {clonal_count} samples out of {clonal_count+mixed_count}!')
else:
    print(
        'Infections are primarily mixed, with {mixed_count} samples out of {clonal_count+mixed_count}!')
