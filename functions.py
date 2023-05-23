# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:08:29 2023

@author: setar
"""

import numpy as np
from sklearn.preprocessing import StandardScaler



def scale_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def replace_missing_values(df):

    df.replace("-", np.nan, inplace=True)
    df_new = df.apply(lambda row: row.fillna(row.mode()[0]), axis=1)

    return df_new


def mixed_infection_classifier(df, col, sparsity_degree):
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