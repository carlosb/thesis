"""This file contains a quick implementation of the
supervised learning algorithm called Laplacian Regularization (LR).
"""
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons


def phi_biharmonic(x, xi, c, d):
    """Returns biharmonic of phi
    """
    result = ((d - norm(x - xi, axis=1)**2) *
              (2 * c**2 * phi(x, xi, c) - phi_laplacian(x, xi, c, d)))
    return result


def phi_laplacian(x, xi, c, d):
    """Returns the laplacian of phi.
    """
    return c * (c * norm(x - xi, axis=1)**2 - d) * phi(x, xi, c)


def phi(x, xi, c):
    """Returns phi
    """
    return np.exp((-c / float(2)) * norm(x - xi, axis=1)**2)


def u(x, w, X, c):
    """Computes u(x).

    Parameters
    ----------
    x : array_like
        Points to predict
    w : array_like
        Weights
    X : array_like
        Dataset
    c : float
        Parameter constant
    """
    return dot(w, phi(x, X, c))


def train(X, y, c, lm=0.5, eta=0.5):
    """Computes the weights given the dataset and targets.

    Parameters
    ----------
    X : array_like
        Matrix of samples
    y : 1-d array
        Target vector
    c : float
        Constant parameter
    lm : float
        Lambda parameter
    eta : float
        Eta parameter
    """
    n = X.shape[0]  # number of samples
    d = X.shape[1]  # number of dimensions

    # Build matrix A
    A = np.zeros((n, n))
    for i in range(n):
        # print phi_biharmonic(X, X[i], c, d)
        if lm == 0:
            A[i] = phi(X, X[i], c)
        else:
            A[i] = phi(X, X[i], c) + lm * phi_biharmonic(X, X[i], c, d)

    # Compute weights
    w = dot(dot(inv(dot(A.T, A) + eta * np.identity(n)), A.T), y)
    return w


# Load dataset
X, y = make_moons(n_samples=400, noise=0.2, random_state=42)

# Preprocess
X = scale(X)

# Train and validate
k = 1
k_cross_validation_mean = 0.
for ki in range(k):
    print 'k =', ki
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create meshgrids
    xl = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 300)
    yl = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 300)
    xx, yy = np.meshgrid(xl, yl)
    Z = np.c_[xx.ravel(), yy.ravel()]

    # Set parameters
    c = 1  # parameter constant
    lm = 0.001  # lambda
    eta = 0.1  # eta

    # Compute weights
    w = train(X_train, y_train, c, lm, eta)

    # Predict meshgrid
    P_mesh = np.zeros((Z.shape[0]))
    for i, x in enumerate(Z):
        # note that we need the dataset which we trained with
        pred = u(x, w, X_train, c)
        P_mesh[i] = pred
    P_mesh[np.where(P_mesh > 0.5)] = 1
    P_mesh[np.where(P_mesh <= 0.5)] = 0
    Z = P_mesh.reshape(xx.shape)

    # Predict test set
    P_test = np.zeros(y_test.shape)
    for i, x in enumerate(X_test):
        # note that we need the dataset which we trained with
        pred = u(x, w, X_train, c)
        P_test[i] = pred

    # Plot
    plt.contour(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    P_test[np.where(P_test > 0.5)] = 1
    P_test[np.where(P_test <= 0.5)] = 0
    acc = np.sum(P_test == y_test) / float(len(X_test)) * 100.
    print '%.2f%s correct with n =' % (acc, '%'), len(X_test)

    k_cross_validation_mean += acc

k_cross_validation_mean /= k
print 'K-Cross Validation score: ', k_cross_validation_mean
plt.title('br')
plt.show()
