# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Closing all the figures
plt.close('all') 

# GENERATING AND PLOTING THE DATASET
X, y = make_circles(noise=0.08, factor=0.2, random_state=42)
X.shape, y.shape
print(y.shape,X.shape)

# Create figure to draw chart
plt.figure(2, figsize=(6, 6))

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')

# Format chart
plt.xlabel('x1')
plt.ylabel('x2')
plt.xticks(())
plt.yticks(())
plt.show()

# LOGISTIC REGRESSION: MODEL TRAINING AND ACCURACY ESTIMATION on X,y dataset


#====================== YOUR CODE HERE =====================================

clf = LogisticRegression()
clf.fit(X,y)

ypred=clf.predict(X)
print(accuracy_score(y,ypred))
print("done")

#===========================================================================


# Boundary visualization

# We create a grid of points contained within
# [x1_min - 0.5, x1_max + 0.5] x [y1_min - 0.5, y1_max + 0.5] with step h=0.02
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size of the grid

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
X_grid = np.c_[xx1.ravel(), xx2.ravel()]

# Retrieve predictions for all data points of the grid in the new variable y_grid
# meaning predict the classifier answer for y_grid data

#====================== YOUR CODE HERE =====================================

clf = LogisticRegression()
clf.fit(X,y)

y_grid = clf.predict(X_grid)

#===========================================================================

#Reshape y_grid to the shape of the grid
y_grid = y_grid.reshape(xx1.shape)
# Create figure to draw chart
plt.figure(2, figsize=(6, 6))

try:
    # Plot the decision boundary (label predicted assigned to a color)
    plt.pcolormesh(xx1, xx2, y_grid, cmap=plt.cm.Set1, shading='auto')
except:
    print("Something went wrong 😣 Are you sure you retrieved predictions \
on your grid points and reshaped it to the proper size?")
    pass

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Set1)

# Format chart
plt.xlabel('x1')
plt.ylabel('x2')
plt.xticks(())
plt.yticks(())
plt.show()

# ADDING NEW FEATURES => X_new = [X1, Z=X1²+X2²] AND  X_new plotting

z = X[:,0]*X[:,0] + X[:,1]*X[:,1]
X_new = np.column_stack((X, z))

# Create figure to draw chart
plt.figure(3, figsize=(6, 6))

# Plot the training points
plt.scatter(X_new[:, 0], X_new[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k')

# Format chart
plt.xlabel('x1')
plt.ylabel('z')
plt.xticks(())
plt.yticks(())
plt.show()
#===========================================================================

# Logistic regression and accuracy on Xnew,y dataset

# ======================= YOUR CODE HERE ====================================

lf = LogisticRegression()
clf.fit(X_new,y)

ypred=clf.predict(X_new)
print(accuracy_score(y,ypred))
cnf_matrix=confusion_matrix(y,ypred)

class_names=[0,1] # name  of classes
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

#=========================================================================