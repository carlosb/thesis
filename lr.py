import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


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


target_cleanup = {'M': 1, 'B': 0}

# Load dataset
df = pd.read_csv('datasets/breast_cancer/WDBC.csv', header=None)

# Remove target column from inputs and extract it
id_column = df.columns[0]
target_column = df.columns[1]
inputs = df.drop([id_column, target_column], axis=1)
targets = df[target_column].replace(target_cleanup)

# Convert to matrix
X = inputs.as_matrix()
y = targets.as_matrix()

X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Set parameters
c = 0.001           # parameter constant
lm = 0.1       # lambda
eta = 0.1      # eta

# Compute weights
w = train(X_train, y_train, c, lm, eta)

# Predict
acc = 0
for i, x in enumerate(X_test):
    # note that we need the dataset which we trained with
    pred = u(x, w, X_train, c)

    # threshold values
    if pred > 0.5:
        pred = 1
    else:
        pred = 0

    # compare
    if pred == y_test[i]:
        acc += 1
acc = acc / float(len(X_test)) * 100.
print acc
