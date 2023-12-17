# -*- coding: utf-8 -*-
"""
Created on 16/09/2022

@author: MARENGO Matteo
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from os.path import join
import matplotlib.pyplot as plt
import csv

plt.close('all')

# DATA LOADING: X is the feature matrix and Y is the label vector

filename = r'creditcard.csv'
dirpath = r'D:\OneDrive\Documents\PostCPGE\PHELMA 2020-2023\BIOMED 2022-2023\Machine Learning\Sequence-1-Bayes-KNN\Datasets'
filepath = join(dirpath, filename)
print(filepath)
print("\n")

file = open(filepath,"r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data))

#add
data = data.astype(float)

X = data[:, 0:30]
Y = data[: , 30]


##############################################################################
# DATA SET REPARTITION

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

print("Normal transaction: ",nzer)
print("Fraudulent transaction: ",nun)

list_random = list(zip(X, Y))
np.random.shuffle(list_random)
X_random, y_random = zip(*list_random)

# Split XTrain and yTrain
XTrain = np.array(X_random[:int(len(X)*0.7)])
yTrain = np.array(y_random[:int(len(X)*0.7)])

# Split XTest and yTest
XTest = np.array(X_random[int(len(X)*0.7):])
yTest = np.array(y_random[int(len(X)*0.7):])
##############################################################################
# GAUSSIAN NAIVE BAEYSIAN CLASSIFIER

# TO BE COMPLETED

gnb = GaussianNB()

gnb.fit(XTrain,yTrain)

Ypredict = gnb.predict(XTest)

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

recall= cfm[1,1]/(cfm[1,1]+cfm[1,0])
precision = cfm[1,1]/(cfm[1,1]+cfm[0,1])

print("Precision: ",precision)
print("Recall: ",recall)


##############################################################################

# K-NN Classification

# TO BE COMPLETED

print("KNN classification with normalization : \n")
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/

# Data normalization

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

print(data)

X = scaled[:, 0:30]
Y = scaled[: , 30]


# GAUSSIAN NAIVE BAEYSIAN CLASSIFIER
# https://blog.floydhub.com/naive-bayes-for-machine-learning/

 # Data conversion towards integer values   
scaled = scaled.astype(float)
X = scaled[:, 0:30]
Y = scaled[: , 30]

# Data splitting: 70% for the training set and 30% for the testing set
# https://www.sharpsightlabs.com/blog/scikit-train_test_split/
# TO BE COMPLETED


list_random = list(zip(X, Y))
np.random.shuffle(list_random)
X_random, y_random = zip(*list_random)

# Split XTrain and yTrain
XTrain = np.array(X_random[:int(len(X)*0.7)])
yTrain = np.array(y_random[:int(len(X)*0.7)])

# Split XTest and yTest
XTest = np.array(X_random[int(len(X)*0.7):])
yTest = np.array(y_random[int(len(X)*0.7):])

# K-NN classification

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# TO BE COMPLETED

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(XTrain,yTrain)

Ypredict = neigh.predict(XTest)

# Accuracy score

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

# TO BE COMPLETED

cfm = confusion_matrix(yTest,Ypredict)
print(cfm)
print(cfm[0,0]," True Negatives")
print(cfm[0,1],"  False Positives")
print(cfm[1,0],"  False Negatives")
print(cfm[1,1],"  True Positives\n")

recall= cfm[1,1]/(cfm[1,1]+cfm[1,0])
precision = cfm[1,1]/(cfm[1,1]+cfm[0,1])

print("Precision: ",precision)
print("Recall: ",recall)
