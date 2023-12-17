# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:58:01 2021

@author: capliera
"""

#imports
import numpy as np
import math
import csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

###############################################################################

# donwload the Boston dataset

file = open('housing.csv',"r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data))
data = data.astype(float)
X = data[:, 0:3]
y = data[: , 3]
nb_sample = len(y)
ind=np.arange(nb_sample)

###############################################################################

# DATASET SPLTING

### TO BE COMPLETED

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)

# BASELINE LINEAR REGRESSION

### TO BE COMPLETED
print("\n")
print("Baseline Linear regression:\n ")

reg = LinearRegression().fit(XTrain,yTrain)
print("Reg Score: ",reg.score(XTrain,yTrain))
print("Reg Coef: ",reg.coef_)

yPredict = reg.predict(XTest)

print("Validation Score: ", reg.score(XTest,yTest))
print("\n")

###############################################################################

# ADDING NORMALIZED POLYNOMIAL FEATURES

### TO BE COMPLETED
print("With Normalized polynomial features:\n")

trans = StandardScaler()
X_train_scaled = trans.fit_transform(XTrain)
X_test_scaled = trans.fit_transform(XTest)
poly = PolynomialFeatures(degree=5)
X_train_scaled_poly = poly.fit_transform(X_train_scaled)
X_test_scaled_poly = poly.fit_transform(X_test_scaled)


# XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)

reg = LinearRegression().fit(X_train_scaled_poly,yTrain)
print("Reg Score: ",reg.score(X_train_scaled_poly,yTrain))
print("Reg Coef: ",reg.coef_)

yPredict = reg.predict(X_test_scaled_poly)

print("Validation Score: ", reg.score(X_test_scaled_poly,yTest))
print("\n")

###############################################################################

# APPLYING REGULARIZATION

### TO BE COMPLETED
print("With Regularization: \n")

clf = Ridge(alpha=0.5)
clf.fit(X_train_scaled_poly,yTrain)
#clf.fit(X_test_scaled_poly,yTest)

# XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)

#reg = LinearRegression().fit(X_train_scaled_poly,yTrain)
print("Reg Score: ",clf.score(X_train_scaled_poly,yTrain))
print("Reg Coef: ",clf.coef_)

yPredict = clf.predict(X_test_scaled_poly)

print("Validation Score: ", clf.score(X_test_scaled_poly,yTest))

