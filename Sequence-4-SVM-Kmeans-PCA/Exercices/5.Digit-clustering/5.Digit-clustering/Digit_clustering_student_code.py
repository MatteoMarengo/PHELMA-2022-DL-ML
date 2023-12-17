# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:30:52 2020

@author: capliera
"""

from sklearn.datasets import load_digits
# Scientific and vector computation for python
import numpy as np
# Kmeans function of the sklearn library
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix


# Digit data set loading
digits = load_digits()

# Dispalying some samples
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


#Proceeding to k-mean clustering
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

#Displaying the final k-means centroids
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    
# Computing k-means clustering performances

# Matching between the index attributed via the kmeans function (cf. clusters table)
# and the real digit number (cf. digits.target table) because the Kmeans algo
# attributes the available label [0, ... K-1] to a given cluster by chance

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
    
print("Accuracy = ", str(round(accuracy_score(digits.target, labels)*100, 2)))
mat = confusion_matrix(digits.target, labels)
plt.figure()
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# Proceeding to k-means clustering with different k-values from 1 to 10

# TO BE COMPLETED