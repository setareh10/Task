# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:35:52 2023

@author: setar
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""




from sklearn.model_selection import cross_validate, KFold
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pathlib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import LinearSVC


def scaling(df):
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


dataset_A = pd.read_csv(pathlib.Path(
    "C:\\Users\\setar\\Desktop\\ML task\\florinov-dc\\dataset-A.tsv"), sep='\t')
dataset_C = pd.read_csv(pathlib.Path(
    "C:\\Users\\setar\\Desktop\\ML task\\florinov-dc\\dataset-C.tsv"), sep='\t')
dataset_D = pd.read_csv(pathlib.Path(
    "C:\\Users\\setar\\Desktop\\ML task\\florinov-dc\\dataset-D.tsv"), sep='\t')
dataset_E = pd.read_csv(pathlib.Path(
    "C:\\Users\\setar\\Desktop\\ML task\\florinov-dc\\dataset-E.tsv"), sep='\t')


dataset_A = dataset_A.drop(['Unnamed: 0'], axis=1)
dataset_C = dataset_C.drop(['Unnamed: 0'], axis=1)
dataset_D = dataset_D.drop(['Unnamed: 0'], axis=1)
dataset_E = dataset_E.drop(['Unnamed: 0'], axis=1)

dataset_A_new = replace_missing_values(dataset_A)
dataset_C_new = replace_missing_values(dataset_C)
dataset_D_new = replace_missing_values(dataset_D)
dataset_E_new = replace_missing_values(dataset_E)


################################# Question1-A #################################

a = pd.get_dummies(dataset_A_new)
pca = PCA(25)
a_prime = pca.fit_transform(a)
print(f'Total explained variance: {(pca.explained_variance_ratio_).sum()}')

a_scaled = scaling(a_prime)

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


################################# Question1-B #################################

count_heterozygous(dataset_A_new)
# mixed_infection(dataset_A_new)

dataset_A_new["mixed_infection"] = "Clonal"

dataset_A_classified = mixed_infection_classifier(
    dataset_A_new, "heterozygous_unique", 3)
infection_nb = dataset_A_classified["mixed_infection"].value_counts()
clonal_count = infection_nb[0]
mixed_count = infection_nb[1]

if clonal_count > mixed_count:
    print('Infections are primarily clonal!')
else:
    print('Infections are primarily mixed!')


################################# Question1-C #################################

c = pd.get_dummies(dataset_C_new)
c_prime = pca.fit_transform(c)

model = KMeans(n_clusters=n_clusters)
model.fit(a_scaled)
c_scaled = scaling(c_prime)

categories = model.predict(c_scaled)
for i in range(categories.shape[0]):

    print('sample ' + str(i) + ' belongs to population ' + str(categories[i]))

################################# Question1-D #################################


dataset_A_mixed = dataset_A_classified[dataset_A_classified['mixed_infection'] == "Mixed"]

a = pd.get_dummies(dataset_A_mixed)
a_prime = pca.fit_transform(a)

d = pd.get_dummies(dataset_D_new)
d_prime = pca.fit_transform(d)

model = KMeans(n_clusters=2)
model.fit(a_scaled)
d_scaled = scaling(d_prime)

categories = model.predict(d_scaled)

for i in range(categories.shape[00]):

    print('sample ' + str(i) + ' belongs to population ' + str(categories[i]))


################################# Question1-E #################################


e = pd.get_dummies(dataset_E_new)
pca = PCA(25)
e_prime = pca.fit_transform(e)
e_scaled = scaling(e_prime)


Z = linkage(e_scaled, method='ward')
fig = plt.figure(figsize=(25, 10))
plt.title('Hierarchical clustering of CheapSeq', fontsize=18)
dn = dendrogram(Z)


x0 = e_scaled.copy()
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 20))
visualizer.fit(x0)
n_clusters = visualizer.elbow_value_

if n_clusters > 1:
    print("Different populations have been identified!")
else:
    print("One single population has been identified!")


count_heterozygous(dataset_E_new)

dataset_E_new["mixed_infection"] = "Clonal"


dataset_E_classified = mixed_infection_classifier(
    dataset_E_new, "heterozygous_unique", 3)
infection_nb = dataset_E_classified["mixed_infection"].value_counts()
clonal_count = infection_nb[0]
mixed_count = infection_nb[1]

if clonal_count > mixed_count:
    print('Infections are primarily clonal!')
else:
    print('Infections are primarily mixed!')


################################# Question1-F #################################

x_A = dataset_A_classified.drop(
    ['heterozygous_unique', 'heterozygous_count', 'mixed_infection'], axis=1)
y_A = dataset_A_classified['mixed_infection']

x_E = dataset_E_classified.drop(
    ['heterozygous_unique', 'heterozygous_count', 'mixed_infection'], axis=1)
y_E = dataset_E_classified['mixed_infection']

x_A_num = pd.get_dummies(x_A)
x_E_num = pd.get_dummies(x_E)

pca = PCA(25)
x_A_num_prime = pca.fit_transform(x_A_num)
x_E_num_prime = pca.fit_transform(x_E_num)

X_A_scaled = scaling(x_A_num_prime)
X_E_scaled = scaling(x_E_num_prime)

n_splits = 5
kf = KFold(n_splits=n_splits)


SVC = LinearSVC()


scores_A = cross_validate(
    SVC,
    X_A_scaled,
    y_A,
    cv=kf,
    n_jobs=1,
)

std_A = scores_A['test_score'].std()
mean_A = scores_A['test_score'].mean()
print(f'Accuracy interval: ({mean_A-std_A}, {mean_A}, {mean_A+std_A})')

scores_E = cross_validate(
    SVC,
    X_E_scaled,
    y_E,
    cv=kf,
    n_jobs=1,
)

std_E = scores_E['test_score'].std()
mean_E = scores_E['test_score'].mean()
print(f'Accuracy interval: ({mean_E-std_E}, {mean_E}, {mean_E+std_E})')
