# -*- coding: utf-8 -*-
"""
Created on 16/09/2022

@author: MARENGO Matteo
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

plt.close('all')

# DATA LOADING: X is the feature matrix and Y is the label vector

filename = r'caesariandata.csv'
dirpath = r'D:\OneDrive\Documents\PostCPGE\PHELMA 2020-2023\BIOMED 2022-2023\Machine Learning\Sequence-1-Bayes-KNN\Datasets'
filepath = join(dirpath, filename)
print(filepath)
print("\n")

file = open(filepath,"r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data))
data = data.astype(int)
X = data[:, 0:1]
Y = data[: , 5]

# DATA REPARTITION

#TO BE COMPLETED

nb_samples = len(Y)
print("Number of samples: ",nb_samples)
nzer = 0
nun = 0

for i in range (nb_samples):
    if Y[i] == 0 :
        nzer = nzer + 1
    if Y[i] == 1 :
        nun = nun + 1

print("Zeros: ",nzer)
print("One: ",nun)

XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Manual Training and Testing sets definition (70% and 30% of the total data set)
# for the first trial and then by using with the train_test_split function for 
# the second trial

# TO BE COMPLETED

cnb = GaussianNB()

cnb.fit(XTrain,yTrain)

Ypredict = cnb.predict(XTest)

# Training and testing Accuracies: computing the % of good answers

# TO BE COMPLETED

errors = 0

for i in range(len(yTest)):
    if yTest[i] != Ypredict[i]:
        errors = errors +1

print("There are" ,errors, "errors")

prop = errors/len(yTest)
propprct = prop*100

print("Percentage of error is:",propprct,"%\n")

print(classification_report(yTest,Ypredict))

# Making the Confusion Matrix
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

#TO BE COMPLETED

cfm = confusion_matrix(yTest,Ypredict)
print(cfm)
print(cfm[0,0]," True Negatives")
print(cfm[0,1],"  False Positives")
print(cfm[1,0],"  False Negatives")
print(cfm[1,1],"  True Positives\n")

# Autatic data splitting for the second trial

# Naive bayes classification

# TO BE COMPLETED

file = open(filepath,"r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data))
data = data.astype(int)
X = data[:, 2:4]
Y = data[:, 5]



# Accuracies and confusion matrix

# TO BE COMPLETED

nb_samples = len(Y)
print("Number of samples: ",nb_samples)
nzer = 0
nun = 0

for i in range (nb_samples):
    if Y[i] == 0 :
        nzer = nzer + 1
    if Y[i] == 1 :
        nun = nun + 1

print("Zeros: ",nzer)
print("One: ",nun)

XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Manual Training and Testing sets definition (70% and 30% of the total data set)
# for the first trial and then by using with the train_test_split function for 
# the second trial

# TO BE COMPLETED

cnb = CategoricalNB()

cnb.fit(XTrain,yTrain)

Ypredict = cnb.predict(XTest)

# Training and testing Accuracies: computing the % of good answers

# TO BE COMPLETED

errors = 0

for i in range(len(yTest)):
    if yTest[i] != Ypredict[i]:
        errors = errors +1

print("There are" ,errors, "errors")

prop = errors/len(yTest)
propprct = prop*100

print("Percentage of error is:",propprct,"%\n")

print(classification_report(yTest,Ypredict))

# Making the Confusion Matrix
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

#TO BE COMPLETED

cfm = confusion_matrix(yTest,Ypredict)
print(cfm)
print(cfm[0,0]," True Negatives")
print(cfm[0,1],"  False Positives")
print(cfm[1,0],"  False Negatives")
print(cfm[1,1],"  True Positives\n")




