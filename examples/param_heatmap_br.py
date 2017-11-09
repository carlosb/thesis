"""This file contains a quick implementation of the
supervised learning algorithm called Laplacian Regularization (LR).
"""
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from matplotlib import pyplot as plt


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
# Dictionary to convert to numeric data
target_cleanup = {'M': 1, 'B': -1}

# Load dataset
# X, y = make_circles(n_samples=400, noise=0.2, random_state=42, factor=0.2)
# X, y = make_moons(n_samples=400, noise=0.2, random_state=42)

df = pd.read_csv('datasets/breast_cancer/WDBC.csv', header=None)
id_column = df.columns[0]
target_column = df.columns[1]
inputs = df.drop([id_column, target_column], axis=1)
targets = df[target_column].replace(target_cleanup)

# Convert to matrix
X = inputs.as_matrix()
y = targets.as_matrix()

# Preprocess
X = scale(X)

eta = 0.1  # eta

# Decision boundary meshgrid
# xl = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
# yl = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
# xx, yy = np.meshgrid(xl, yl)
# zz = np.c_[xx.ravel(), yy.ravel()]

# Parameter meshgrid
c_vals = np.arange(0, 1, 0.1)
lm_vals = np.arange(0, 1, 0.1)
cc, ll = np.meshgrid(c_vals, lm_vals)
hh = np.c_[cc.ravel(), ll.ravel()]

# We are going to plot 4 subplots
f, axarr = plt.subplots(2, 2)

# # Plot dataset
# axarr[0, 0].scatter(X[:, 0], X[:, 1], c=y)
# axarr[0, 0].set_title('dataset')

# Train and validate
k = 1
for ki in range(k):
    print 'k = %d of %d \n%s' % (ki + 1, k, '============')
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    H = []
    it = 0
    max_acc = 0
    for c, lm in hh:
        # Display iteration info
        print it, ' of ', len(hh)
        print '(c, lm) = (%f, %f)' % (c, lm)

        # Compute weights
        w = train(X_train, y_train, c, lm, eta)

        # Predict test set
        P_test = np.zeros(y_test.shape)
        for i, xi in enumerate(X_test):
            # Make the prediction.
            # (We need the training dataset to predict)
            pred = u(xi, w, X_train, c)
            P_test[i] = pred
        P_test[np.where(P_test > 0.5)] = 1
        P_test[np.where(P_test <= 0.5)] = -1
        acc = np.sum(P_test == y_test) / float(len(X_test))
        print 'Accuracy: ', acc

        # Keep track of max accuracy
        if max_acc < acc:
            max_acc = acc
            best_c = c
            best_w = w

        # Append the acc so we get f(c, lm) = z = acc
        H.append(acc)
        it += 1

    print 'Max acc for k = %d was %f\n%s' % (ki, max_acc, '============')

    # # Make the prediction over the meshgrid using best params
    # P_mesh = np.zeros(len(zz))
    # for i, xi in enumerate(zz):
    #     # Make the prediction.
    #     # (We need the training dataset to predict)
    #     pred = u(xi, best_w, X_train, best_c)
    #     P_mesh[i] = pred

    # # Plot contours
    # Z = P_mesh.reshape(xx.shape)
    # axarr[0, 1].contourf(xx, yy, Z)
    # axarr[0, 1].set_title('Contour plot of prediction surface')

    # # Plot Decision boundary
    # P_mesh[np.where(P_mesh > 0.5)] = 1
    # P_mesh[np.where(P_mesh <= 0.5)] = 0

    # Z = P_mesh.reshape(xx.shape)
    # axarr[1, 0].scatter(zz[:, 0], zz[:, 1], c=y)
    # axarr[1, 0].contour(xx, yy, Z)
    # axarr[1, 0].set_title('decision boundary')

    # Plot parameter heatmap
    H = np.array(H)
    H = H.reshape(cc.shape)
    axarr[1, 1].contourf(cc, ll, H)
    axarr[1, 1].set_title('(c, lambda) ---> acc')
    axarr[1, 1].set_xlabel('c')
    axarr[1, 1].set_ylabel('lambda')

    # Set title displaying max accuracy
    f.suptitle('Biharmonic Regularization: %.2f%%' % (max_acc * 100),
               fontweight='bold')
    plt.show()
