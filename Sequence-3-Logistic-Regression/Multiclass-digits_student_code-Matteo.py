# -*- coding: utf-8 -*-

"""
Created on Sun Jun 27 16:27:09 2021

@author: capliera

"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#Figure closing
plt.close('all')

# Dataset loading and representation

digits_df = datasets.load_digits()
# print(digits_df.DESCR)
print('Digits dataset structure= ', dir(digits_df))
print('Data shape= ', digits_df.data.shape)
print('Data contains pixel representation of each image, \n', digits_df.data)

# Using subplot to plot the digits from 0 to 4
rows = 1
columns = 5
fig, ax =  plt.subplots(rows, columns, figsize = (15,6))

plt.gray()
for i in range(columns):
  ax[i].matshow(digits_df.images[i]) 
  ax[i].set_title('Label: %s\n' % digits_df.target_names[i])
  
plt.show()

X = digits_df.data
y = digits_df.target

print("\n")
print("Numper of samples per class:\n")


for i in range(0,10):
    tot = 0
    for j in range(len(y)):
        if y[j]==i:
            tot=tot+1
    print(tot)
    
print("\n")
    

#=========================== YOUR CODE HERE ==================================

# Multiclass Logistic regression: one versus all (parameter multi_class= OVR)

# Data splitting

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the model

clf = LogisticRegression(C=1,max_iter=2900,multi_class='ovr')
clf.fit(XTrain,yTrain)

ypredtrain = clf.predict(XTrain)
ypredtest = clf.predict(XTest)

# Testing the model on the 200th image


print("Prediction of the 200th sample:\n",ypredtest[200]) 
print(y[200])
print("\n")

# Model performances analysis: training and testing accuracies 

print("Train score:",clf.score(XTrain,yTrain))
print("Test score:",clf.score(XTest,yTest))
# print("Train score:",accuracy_score(yTrain,ypredtrain))

# Making the Confusion Matrix

cnf_matrix = confusion_matrix(yTest, ypredtest)
print(cnf_matrix)
print("\n")

class_names=[0,1,2,3,4,5,6,7,8,9] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Classification report

class_names=['0','1','2','3','4','5','6','7','8','9'] # name  of classes
print(classification_report(yTest,ypredtest,target_names=class_names))
a = cnf_matrix.diagonal()/cnf_matrix.sum(axis=1)
for i in range(0,10):
    print(a[i])
    
report = classification_report(yTest,ypredtest,target_names=class_names,output_dict=True)
df = pandas.DataFrame(report).transpose()

#===========================================================================
