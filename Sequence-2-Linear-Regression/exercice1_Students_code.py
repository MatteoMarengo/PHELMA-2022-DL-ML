# -*- coding: utf-8 -*-

# Exercise 1: food truck profit prediction with  linear regression function
# predicted_ profits = theta0 + theta1 * town_population
#Goal : to estimate the best values for theta0 and theta1 parameters

# Scientific and vector computation for python
import numpy as np

from os.path import join


# Plotting library
from matplotlib import pyplot


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data 
    points and gives the figure axes labels of population and profit.
    """    
    
    pyplot.figure()  # open a new figure
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')

def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost J of using theta as the
    parameter for linear regression to fit the data points in X and y.
    
    """
    # initialize some useful values
    m = y.size  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
        
    # ====================== YOUR CODE HERE =====================
   
    for i in range(m):
        J=J+(theta[0]+theta[1]*X[i,1]-y[i])**2
    
    J=J/(2*m)   
    
    # ===========================================================
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
   """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        # j = 0
        dj0=0
        for j in range(m):
            dj0 = dj0 + theta[0]+theta[1]*X[j,1]-y[j]
        dj0 = dj0/m
        
        # j = 1
        dj1=0
        for j in range(m):
            dj1 = dj1 + (theta[0]+theta[1]*X[j,1]-y[j])*X[j,1]
        dj1 = dj1/m
        
        theta[0]=theta[0]-alpha*dj0
        theta[1]=theta[1]-alpha*dj1
        
        # =====================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history

# LINEAR REGRESSION for food truck opening prediction
# ===================================================
# Data set reading and display
"""
filename = r'ex1data1.txt'
dirpath = r"D:\OneDrive\Documents\PostCPGE\PHELMA 2020-2023\BIOMED 2022-2023\Machine Learning\Sequence-2-Linear-Regression\Student-codes-and-data"
filepath = join(dirpath, filename)
data = np.loadtxt(filepath)
"""
data = np.loadtxt('ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size  # number of training examples

plotData(X, y)

# Add a column of ones to X (the intercept term). The numpy function stack 
# joins arrays along a given axis. 
# The first axis (axis=0) refers to rows (training examples) 
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(m), X], axis=1)

# Cost computation with specific theta values

J = computeCost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')

# Gradient descent process

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

# Plot the cost function evolution
pyplot.figure()
pyplot.title("Cost")
pyplot.ylim(top=5.8)
pyplot.ylim(bottom=4.4)
pyplot.plot(J_history)


# plot the linear estimated fit
plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))


