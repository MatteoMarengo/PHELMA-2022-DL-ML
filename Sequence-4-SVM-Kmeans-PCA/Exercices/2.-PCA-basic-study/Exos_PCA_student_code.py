# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:40:26 2020

@author: capliera
"""

import matplotlib.pyplot as plt # graph plots
import numpy as np 
import pandas as pd

from sklearn import datasets # datasets 
from sklearn import preprocessing # data normalization
from sklearn import decomposition # PCA

from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.svm import SVC

import utils


np.random.seed = 2017 # for reproductible results

def graph_acp2(X_PC2, y): # For data display after PCA

    plt.figure(figsize=(15,4))
    plt.subplot(1, 3, 1)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.scatter(X_PC2[:, 0], X_PC2[:, 1], c=y)
    plt.subplot(1, 3, 2)
    plt.title("Dimension of the highest variance :")
    plt.scatter(X_PC2[:, 0], np.ones(X_PC2.shape[0]), c=y)
    plt.subplot(1, 3, 3)
    plt.title("Dimension of the 2nd variance :")
    plt.scatter(X_PC2[:, 1], np.ones(X_PC2.shape[0]), c=y)
    plt.show()

# Dataset loading and display

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

print("Data dimension: {}".format(X_iris.shape[1]))

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.title("Sepal dimensions 0 and 1:")
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris)
plt.subplot(1, 2, 2)
plt.title("Petal dimensions 2 and 3:")
plt.scatter(X_iris[:, 2], X_iris[:, 3], c=y_iris)
plt.plot([1,7],[0.05,2.5])
plt.show()

# PCA (call X_iris_PCA the data feature matrix after PCA transform)

# TO BE COMPLETED

trans = StandardScaler()
X_iris_scaled = trans.fit_transform(X_iris)
pca = PCA(n_components=2)
X_iris_PCA = pca.fit_transform(X_iris_scaled)


# Linear SVM classification of the iris dataset

# TO BE COMPLETED

clf = SVC(C=100,kernel='linear')
clf.fit(X_iris_PCA,y_iris)


# PCA counter examples

# Sometimes, depending on the data, the discrimination is 
# not related to the highest component - Lookat the following example.

X11 = np.random.rand(30)*10
X21 = X11 + 1
X12 = np.random.rand(20)*10
X22 = X12 + 2
X = np.array([np.concatenate((X11,X12)),
              np.concatenate((X21,X22))]).T
y = np.concatenate((np.zeros(30), np.ones(20)))
X = preprocessing.scale(X, with_mean=True, with_std=True)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# In this case the second component is more efficient to discriminate the data

pca = decomposition.PCA(n_components=2)
X_PC2 = pca.fit(X).transform(X)
graph_acp2(X_PC2, y)

# When the data have homogeneous variances, PCA is useless when the data variance
# is similar along each intial dimensions. Look at the following example

X11 = np.random.normal(0, 10, 500)
X21 = abs(np.random.normal(0, 10, 500))
X12 = np.random.normal(0, 10, 500)
X22 = -abs(np.random.normal(0, 10, 500))
X = np.array([np.concatenate((X11,X12)),
              np.concatenate((X21,X22))]).T
y = np.concatenate((np.zeros(500), np.ones(500)))
y = y.astype(int)
plt.figure(figsize=(6,6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_PC2 = pca.transform(X)
graph_acp2(X_PC2, y)