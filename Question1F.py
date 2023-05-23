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


dataset_A_classified, clonal_count, mixed_count = classifier(
    dataset_A_new, sparsity_degree=3)

dataset_E_classified, clonal_count, mixed_count = classifier(
    dataset_E_new, sparsity_degree=3)


x_A = dataset_A_classified.drop(
    ['heterozygous_unique', 'heterozygous_count', 'mixed_infection'], axis=1)
y_A = dataset_A_classified['mixed_infection']

x_E = dataset_E_classified.drop(
    ['heterozygous_unique', 'heterozygous_count', 'mixed_infection'], axis=1)
y_E = dataset_E_classified['mixed_infection']

pca_a, a_scaled = data_preprocessing(x_A, PCA, n_components=25)
pca_e, e_scaled = data_preprocessing(x_E, PCA, n_components=25)

def linearSVC_classifier (X, y, n_splits):

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
    return mean, std


mean_A, std_A = linearSVC_classifier(a_scaled, y_A, n_splits=5)
mean_E, std_E = linearSVC_classifier(e_scaled, y_E, n_splits=5)

print(f'Accuracy for dataset_A: mean = {round(mean_A,3)}, std = {round(std_A,3)})')

print(f'Accuracy for dataset_E: mean = {round(mean_E,3)}, std = {round(std_E,3)})')
