# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:05:37 2020

@author: capliera
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # graph plots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Import scikit-learn dataset library
from sklearn import datasets

# Import train_test_split function
from sklearn.model_selection import train_test_split

#Import svm model
from sklearn import svm

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics



#LOADING THE DATA
#================
cancer = datasets.load_breast_cancer()

# print the names of the 13 main features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("\n")
print("Labels: ", cancer.target_names)

print("Data set dimensions: ", cancer.data.shape)
print("Values: [0,1]")

X,y = datasets.load_breast_cancer(return_X_y=True)



# DATASET ANALYSIS
#=================

# Counting the positive and négative samples

positives = list(filter(lambda element: element == 1, y))
negatives = list(filter(lambda element: element == 0, y))
print('Positive case: %d' %len(positives))
print('Negative case: %d' %len(negatives))

# Feature Scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# PCA analysis
#%%

pca = PCA(.99) 
pca.fit(X_train)
print(pca.n_components_)

# SVM CLASSIFICATION
#%% LINEAR
svc_model = svm.SVC(kernel='linear')
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_train)
               
print(svc_model.score(X_train, y_train))
print(metrics.accuracy_score(y_train, y_pred))
print(metrics.precision_score(y_train, y_pred))
print(metrics.recall_score(y_train, y_pred))


#%% GAUSSIAN
svc_model = svm.SVC(kernel='rbf')
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_train)
               
print(svc_model.score(X_train, y_train))
print(metrics.accuracy_score(y_train, y_pred))
print(metrics.precision_score(y_train, y_pred))
print(metrics.recall_score(y_train, y_pred))


#%% LINEAR WITH NORMALIZATION
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

svc_model = svm.SVC(kernel='linear')
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_train)
               
print(svc_model.score(X_train, y_train))
print(metrics.accuracy_score(y_train, y_pred))
print(metrics.precision_score(y_train, y_pred))
print(metrics.recall_score(y_train, y_pred))

#%% GAUSSIAN WITH NORMALIZATION
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

svc_model = svm.SVC(kernel='rbf')
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_train)
               
print(svc_model.score(X_train, y_train))
print(metrics.accuracy_score(y_train, y_pred))
print(metrics.precision_score(y_train, y_pred))
print(metrics.recall_score(y_train, y_pred))
#%%
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)