# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:05:37 2020

@author: capliera
"""



import numpy as np
import matplotlib.pyplot as plt # graph plots

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

# print data(feature)shapeprint("Data set dimensions: ", cancer.data.shape)

# DATASET ANALYSIS
#=================

# Counting the positive and négative samples

# TO BE COMPLETED


# Feature Scaling

# TO BE COMPLETED

# PCA analysis

# TO BE COMPLETED

# SVM CLASSIFICATION

# TO BE COMPLETED