# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:10:29 2021

@author: capliera
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
# Plotting library
from matplotlib import pyplot


# Data loading

X, y = load_breast_cancer(return_X_y=True)

# TO DO:  Data description

m=np.size(y)

zer = 0
one = 0

for i in range(m):
    if y[i] == 0:
        zer=zer+1
    else:
        one=one+1
        
print("Number of samples:",m)
print("Number of malignant:",zer)
print("Number of benign:",one) 

# TODO : Logistic regression and accuracy

# Data splitting and Scaling

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Logistic regression   

clf = LogisticRegression(C=1, max_iter=2900)
clf.fit(XTrain,yTrain)

ypredtrain = clf.predict(XTrain)
ypredtest = clf.predict(XTest)

# Accuracy computation

#train_score = clf.score(XTrain,yTrain)
#test_score = clf.score(XTest,yTest)
print(accuracy_score(yTrain, ypredtrain))
print(accuracy_score(yTest,ypredtest))

# NORMALIZATION

trans = StandardScaler()
X_train_scaled = trans.fit_transform(XTrain)
X_test_scaled = trans.fit_transform(XTest)

# Logistic regression   

clf = LogisticRegression(C=1, max_iter=2900)
clf.fit(X_train_scaled,yTrain)
ypredtrain = clf.predict(X_train_scaled)
ypredtest = clf.predict(X_test_scaled)

# Accuracy computation

#train_score = clf.score(XTrain,yTrain)
#test_score = clf.score(XTest,yTest)
print(accuracy_score(yTrain, ypredtrain))
print(accuracy_score(yTest,ypredtest))

# TODO : fit Logistic Regression models with varying values of C

x=[]
y=[]

for i in range(1,41,1):
    clf = LogisticRegression(C=(i/10), max_iter=6000)
    clf.fit(XTrain,yTrain)
    
    x.append(clf.score(XTrain,yTrain))
    y.append(clf.score(XTest,yTest))
   
pyplot.figure()
pyplot.plot(range(1,41),x)
pyplot.plot(range(1,41),y)

pyplot.show()
    
    
    

