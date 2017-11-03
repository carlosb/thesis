"""This file contains a quick implementation of the
supervised learning algorithm called Laplacian Regularization (LR).
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import inv, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.datasets import make_moons, make_circles, make_regression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


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
        A[i] = phi(X, X[i], c) - lm * phi_laplacian(X, X[i], c, d)

    # Compute weights
    w = dot(dot(inv(dot(A.T, A) + eta * np.identity(n)), A.T), y)
    return w


def predict(x, cs, w, c):
    return u(x, w, cs, c)


# Dictionary to convert to numeric data
target_cleanup = {2: 1, 1: 0}

# Load dataset
df = pd.read_csv('../datasets/heart/heart.dat', sep=None, header=None)

# Remove target column from inputs and extract it
inputs = df.drop([len(df.columns) - 1], axis=1)
targets = df[len(df.columns) - 1].replace(target_cleanup)

# Convert to matrix
X = inputs.as_matrix()
y = targets.as_matrix()

# X, y = make_moons(n_samples=400, noise=0.2, random_state=16)
# X, y = make_circles(n_samples=400, noise=0.2, factor=0.3, random_state=100)
X, y = make_regression(n_samples=400, n_features=1, n_informative=2, noise=10.0)

# Preprocess
# X = scale(X)

k = 1
k_cross_validation_mean = 0.

for ki in range(k):
    print 'k =', ki
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Set parameters
    c = 0.01      # parameter constant
    lm = 0.1       # lambda
    eta = 1      # eta

    # Compute weights
    w = train(X_train, y_train, c, lm, eta)

    # Predict train set
    # print 'Evaluating model over train set...'
    # P_train = np.zeros(y_train.shape)
    # for i, x in enumerate(X_train):
    #     pred = predict(x, X_train, w, c)
    #     P_train[i] = pred
    # P_train[np.where(P_train > 0.5)] = 1
    # P_train[np.where(P_train <= 0.5)] = -1
    # acc = np.sum(P_train == y_train) / float(len(X_train)) * 100.
    # print '%.2f%s correct with n = %d\n' % (acc, '%', len(X_train))

    # # Predict test dataset
    # print 'Evaluating model over the test set...'
    # P_test = np.zeros(y_test.shape)
    # for i, x in enumerate(X_test):
    #     pred = predict(x, X_train, w, c)
    #     P_test[i] = pred
    # P_test[np.where(P_test > 0.5)] = 1
    # P_test[np.where(P_test <= 0.5)] = -1
    # acc = np.sum(P_test == y_test) / float(len(X_test)) * 100.
    # print '%.2f%s correct with n = %d\n' % (acc, '%', len(X_test))

    # # Predict whole dataset
    # P_whole = np.zeros(y.shape)
    # print 'Evaluating model over the whole set...'
    # P_whole = np.zeros(y.shape)
    # for i, x in enumerate(X):
    #     pred = predict(x, X_train, w, c)
    #     P_whole[i] = pred
    # P_whole[np.where(P_whole > 0.5)] = 1
    # P_whole[np.where(P_whole <= 0.5)] = -1
    # acc = np.sum(P_whole == y) / float(len(X)) * 100.
    # print '%.2f%s correct with n = %d\n' % (acc, '%', len(X))

    # print 'ROC AUC: ', roc_auc_score(y_test, P_test)

    # k_cross_validation_mean += acc

# k_cross_validation_mean /= k
# print 'K-Cross Validation ROCscore: ', k_cross_validation_mean

# Perform dim reduction
dim = X.shape[1]
if dim > 2:
    print 'Reducing dimensionality for plotting...'
    pca = PCA(2)
    X = pca.fit_transform(X)

# Create meshgrids
# print 'Creating meshgrids for contour plotting...'
# resolution = 1
# xl = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
# yl = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
# xx, yy = np.meshgrid(xl, yl)
# Z = np.c_[xx.ravel(), yy.ravel()]

# # Predict meshgrid
# print 'Predicting over meshgrid...'
# P_mesh = np.zeros((Z.shape[0]))
# for i, x in enumerate(Z):
#     # note that we need the dataset which we trained with
#     if dim > 2:
#         pred = predict(pca.inverse_transform(x), X_train, w, c)
#     else:
#         pred = predict(x, X_train, w, c)
#     P_mesh[i] = pred
# # P_mesh[np.where(P_mesh > 0.5)] = 1
# # P_mesh[np.where(P_mesh <= 0.5)] = 0
# Z = P_mesh.reshape(xx.shape)

# Plot
print 'Plotting meshgrid...'
# plt.contourf(xx, yy, Z)
# plt.contour(xx, yy, Z, cmap='magma')

print 'Plotting scatter plot...'
# plt.scatter(X[:, 0], X[:, 1], c=y)
xx = np.linspace(X.min(), X.max(), 100)
R = np.zeros(xx.shape)
for i, xi in enumerate(xx):
    R[i] = predict(xi, X_train, w, c)

plt.scatter(X, y, c=y)
plt.plot(xx, R, c='black', linewidth=4.0)

plt.title('Regression')
plt.show()
