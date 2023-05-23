# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:06:55 2023

@author: setareh
"""
from functions import data_preprocessing 
from sklearn.model_selection import cross_validate, KFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from Question1B import classifier
from datasets import dataset_A_new, dataset_E_new

## Perform classification on the two datasets
dataset_A_classified, clonal_count, mixed_count = classifier(
    dataset_A_new, sparsity_degree=3)

dataset_E_classified, clonal_count, mixed_count = classifier(
    dataset_E_new, sparsity_degree=3)

## Prepare X and Y for another classification problem
x_A = dataset_A_classified.drop(
    ['heterozygous_unique', 'heterozygous_count', 'mixed_infection'], axis=1)
y_A = dataset_A_classified['mixed_infection']

x_E = dataset_E_classified.drop(
    ['heterozygous_unique', 'heterozygous_count', 'mixed_infection'], axis=1)
y_E = dataset_E_classified['mixed_infection']

## Prepare the data
pca_a, a_scaled = data_preprocessing(x_A, PCA, n_components=25)
pca_e, e_scaled = data_preprocessing(x_E, PCA, n_components=5)

def linearSVC_classifier (X, y, n_splits):
    """
    Apply linear SVM classifier

    Parameters
    ----------
    X : DataFrame.
    y : DataFrame.
    n_splits : int.

    Returns
    -------
    mean: float64.
    std: float64.
    min: float64.
    max: float64.

    """
    kf = KFold(n_splits=n_splits)
    SVC = LinearSVC()
    scores = cross_validate(
        SVC,
        X,
        y,
        cv=kf,
        n_jobs=1,
    )
    
    std = scores['test_score'].std()
    mean = scores['test_score'].mean()
    Min = scores['test_score'].min()
    Max = scores['test_score'].max()

    return mean, std, Min, Max


mean_A, std_A, min_A, max_A = linearSVC_classifier(a_scaled, y_A, n_splits=5)
mean_E, std_E, min_E, max_E = linearSVC_classifier(e_scaled, y_E, n_splits=5)

print(f'Accuracy for dataset_A: mean = {round(mean_A,3)}, std = {round(std_A,3)}')
print(f'Accuracy for dataset_E: mean = {round(mean_E,3)}, std = {round(std_E,3)}')
print('**************************************************')
print(f'Min and Max for dataset_A: min = {round(min_A,3)}, max = {round(max_A,3)}')
print(f'Min and Max for dataset_E: min = {round(min_E,3)}, max = {round(max_E,3)}')
