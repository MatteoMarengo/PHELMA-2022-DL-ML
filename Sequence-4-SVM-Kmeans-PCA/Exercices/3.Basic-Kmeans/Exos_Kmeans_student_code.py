# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 07:46:10 2020

@author: capliera
"""
import sys
import numpy as np
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

sys.path.append('..')


def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data in a nice grid.

    Parameters
    ----------
    X : array_like
        The input data of size (m x n) where m is the number of examples and n is the number of
        features.

    example_width : int, optional
        THe width of each 2-D image in pixels. If not provided, the image is assumed to be square,
        and the width is the floor of the square root of total number of pixels.

    figsize : tuple, optional
        A 2-element tuple indicating the width and height of figure in inches.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')


def featureNormalize(X):
    """
    Normalizes the features in X returns a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    Parameters
    ----------
    X : array_like
        An dataset which is a (m x n) matrix, where m is the number of examples,
        and n is the number of dimensions for each example.

    Returns
    -------
    X_norm : array_like
        The normalized input dataset.

    mu : array_like
        A vector of size n corresponding to the mean for each dimension across all examples.

    sigma : array_like
        A vector of size n corresponding to the standard deviations for each dimension across
        all examples.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma


def plotProgresskMeans(i, X, centroid_history, idx_history):
    """
    A helper function that displays the progress of k-Means as it is running. It is intended for use
    only with 2D data. It plots data points with colors assigned to each centroid. With the
    previous centroids, it also plots a line between the previous locations and current locations
    of the centroids.

    Parameters
    ----------
    i : int
        Current iteration number of k-means. Used for matplotlib animation function.

    X : array_like
        The dataset, which is a matrix (m x n). Note since the plot only supports 2D data, n should
        be equal to 2.

    centroid_history : list
        A list of computed centroids for all iteration.

    idx_history : list
        A list of computed assigned indices for all iterations.
    """
    K = centroid_history[0].shape[0]
    pyplot.gcf().clf()
    cmap = pyplot.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=2)

    for k in range(K):
        current = np.stack([c[k, :] for c in centroid_history[:i+1]], axis=0)
        pyplot.plot(current[:, 0], current[:, 1],
                    '-Xk',
                    mec='k',
                    lw=2,
                    ms=10,
                    mfc=cmap(norm(k)),
                    mew=2)

        pyplot.scatter(X[:, 0], X[:, 1],
                       c=idx_history[i],
                       cmap=cmap,
                       marker='o',
                       s=8**2,
                       linewidths=1,)
    pyplot.grid(False)
    pyplot.title('Iteration number %d' % (i+1))


def runkMeans(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):
    """
    Runs the K-means algorithm.

    Parameters
    ----------
    X : array_like
        The data set of size (m, n). Each row of X is a single example of n dimensions. The
        data set is a total of m examples.

    centroids : array_like
        Initial centroid location for each clusters. This is a matrix of size (K, n). K is the total
        number of clusters and n is the dimensions of each data point.

    findClosestCentroids : func
        A function (implemented by student) reference which computes the cluster assignment for
        each example.

    computeCentroids : func
        A function(implemented by student) reference which computes the centroid of each cluster.

    max_iters : int, optional
        Specifies the total number of interactions of K-Means to execute.

    plot_progress : bool, optional
        A flag that indicates if the function should also plot its progress as the learning happens.
        This is set to false by default.

    Returns
    -------
    centroids : array_like
        A (K x n) matrix of the computed (updated) centroids.
    idx : array_like
        A vector of size (m,) for cluster assignment for each example in the dataset. Each entry
        in idx is within the range [0 ... K-1].

    anim : FuncAnimation, optional
        A matplotlib animation object which can be used to embed a video within the jupyter
        notebook. This is only returned if `plot_progress` is `True`.
    """
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)

        centroids = computeCentroids(X, idx, K)

    if plot_progress:
        fig = pyplot.figure()
        anim = FuncAnimation(fig, plotProgresskMeans,
                             frames=max_iters,
                             interval=500,
                             repeat_delay=2,
                             fargs=(X, centroid_history, idx_history))
        return centroids, idx, anim

    return centroids, idx


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise
#import utils

from matplotlib import pyplot

from sklearn.cluster import KMeans


def findClosestCentroids(X, centroids):
    """
    Computes the centroid memberships for every example.
    
    Parameters
    ----------
    X : array_like
        The dataset of size (m, n) where each row is a single example. 
        That is, we have m examples each of n dimensions.
        
    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the data dimension.
    
    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset X.
    """
    
    # Set K
    #K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)

    # ====================== Closest centroid computation ======================

    for i in np.arange(idx.size):
        
        J = np.sqrt(np.sum(np.square(X[i] - centroids), axis = 1))
            
        idx[i] = np.argmin(J)
    
    # =============================================================
    return idx

def computeCentroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.
    
    Parameters
    ----------
    X : array_like
        The datset where each row is a single data point. That is, it 
        is a matrix of size (m, n) where there are m datapoints each
        having n dimensions. 
    
    idx : array_like 
        A vector (size m) of centroid assignments (i.e. each entry in range [0 ... K-1])
        for each example.
    
    K : int
        Number of clusters
    
    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data 
        points assigned to it.
    
    How it is working
    ------------
    For every centroid, compute mean of all points that
    belong to it. Concretely, the row vector centroids[i, :]
    contain the mean of the data points assigned to
    cluster i.

    """
    # Useful variables
    m, n = X.shape
    centroids = np.zeros((K, n))

    # ====================== Centroid computation ======================

    for i in np.arange(K):
        centroids[i] = np.mean(X[idx == i], axis = 0)    
    
    # =============================================================
    return centroids

def kMeansInitCentroids(X, K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x.
    
    Parameters
    ----------
    X : array_like 
        The dataset of size (m x n).
    
    K : int
        The number of clusters.
    
    Returns
    -------
    centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).
    
    """
    m, n = X.shape
    
    centroids = np.zeros((K, n))

    # ====================== Random centroid intialization ======================

    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    
    # =============================================================
    return centroids

# DATA SET LOADING
data = loadmat("D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/Machine Learning/Sequence-4-SVM-Kmeans-PCA/Exercices/3.Basic-Kmeans/Data/ex7data2.mat")
X = data['X']
pyplot.scatter(X[:, 0], X[:, 1], s=50);
pyplot.title('Data training set')

## RUNNING THE FIRST ITERATION OF KMEANS ALGO STEP BY STEP

# Before starting Kmeans clustering: Set the centroid number K and 
# select an initial set of centroids

K = 3   # 3 Centroids => the label of each sample might be 0, 1, ... K-1
#initial_centroids = np.array([[3, 3], [6, 2], [8, 5]]) # 3 samples of the dataset 
                                                       #are selected as initial centroids

#initial_centroids=kMeansInitCentroids(X,K=3)
print("Initial centroids :\n ", initial_centroids)

# Kmean algo first step: Assign each training sample to its closest centroid
idx = findClosestCentroids(X, initial_centroids)
print('Closest centroids for the first 3 training examples:')
print(idx[:3])
print("\n")

# Kmean algo second step: Recompute the mean of each centroid using the  
# points assigned to it.
centroids = computeCentroids(X, idx, K)
print('Centroids computed after the assignment step:')
print(centroids)
print("\n")

## RUNNING A KMEANS ALGORITHM WITH DIFFERENT CENTROIDS INITIALIZATIONS
# That's mean several iterations of the two basic previous steps

# Settings for running K-Means
K = 3
max_iters = 10

# Centroids initialization: either manually tuned or randomly selected 

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

print("Home made Kmeans algo\n")
print("Initial centroids :\n ", initial_centroids)

# Run K-Means algorithm. The 'true' at the end tells the function to plot
# the progress of K-Means
centroids, idx, anim = runkMeans(X, initial_centroids,
                                       findClosestCentroids, computeCentroids, max_iters, True)
anim
print("Final centroids :\n", centroids)
print("\n")

# USING THE SKLEARN KMeans function

# TO BE COMPLETED

kmeans = KMeans(n_clusters=3,random_state=0).fit(X)

