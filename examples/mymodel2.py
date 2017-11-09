"""
My model with biharmonic regularization.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter


def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def sigmoid_derivative(x):
    s = sigmoid(x)
    return (1 - s) * s


def phi(x, ci, c):
    """Gaussian Radial Basis Kernel.
    x : array_like
        Point.
    ci : array_like
        Center of RBF kernel.
    c : float
        Consant parameter.
    """
    return np.exp((-c / float(2)) * np.linalg.norm(x - ci, axis=1)**2)


def grad_phi(x, ci, c):
    return -c * ((x - ci).T * phi(x, ci, c)).T


def laplacian_phi(x, ci, c):
    return c * (c * np.linalg.norm(x - ci, axis=1)**2 - len(x)) * phi(x, ci, c)


def biharmonic_phi(x, ci, c):
    """Returns biharmonic of phi
    """
    result = ((len(x) - np.linalg.norm(x - ci, axis=1)**2) *
              (2 * c**2 * phi(x, ci, c) - laplacian_phi(x, ci, c)))
    return result


def u(x, cs, w, c):
    """Radial Basis Function Approximation.

    Parameters
    ----------
    x : array_like
        Points to predict
    w : array_like
        Weights
    cs : array_like
        Centers of RBFs
    c : float
        Parameter constant
    """
    return np.dot(w, phi(x, cs, c))


def grad_u(x, cs, w, c):
    return np.dot(w, grad_phi(x, cs, c))


def laplacian_u(x, cs, w, c):
    """Laplacian of u.
    """
    return np.dot(w, laplacian_phi(x, cs, c))


def biharmonic_u(x, cs, w, c):
    return np.dot(w, biharmonic_phi(x, cs, c))


def gradw_u(x, cs, c):
    """Gradient of the RBF Approximation with respect to the weights.
    """
    return phi(x, cs, c)


def g(x, cs, w, c):
    return np.sum(((x - cs).T * (phi(x, cs, c) * w)).T)


def grad_f(x, cs, w, c, lm):
    ds = sigmoid_derivative(u(x, cs, w, c))
    gwu = gradw_u(x, cs, c)
    # return ds * gwu + lm * biharmonic_phi(x, cs, c) + lm * phi(x, cs, c)
    return ds * gwu + lm * biharmonic_phi(x, cs, c)


def f(x, cs, w, c, lm):
    # return sigmoid(u(x, cs, w, c)) + lm * biharmonic_u(x, cs, w, c) + u(x, cs, w, c)
    return sigmoid(u(x, cs, w, c)) + lm * biharmonic_u(x, cs, w, c)


def predict(x, cs, w, c):
    return sigmoid(u(x, cs, w, c))


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

X, y = make_moons(n_samples=1000, noise=0.2, random_state=16)

# Preprocess
X = scale(X)

# Set parameters
# c = 0.2    # parameter constant
# lm = 10  # lambda
# eta = 10  # eta
c = 5    # parameter constant
lm = 1  # lambda
eta = 10  # eta

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Compute weights using Levenberg-Marquardt
print 'Training model...'
max_it = 20
n = len(X_train)
# w = np.random.random(n)
w = np.zeros(n)
iterations = 0
eps = 1e-4
while iterations < max_it:
    J = np.zeros((n, n))
    fB = np.zeros(y_train.shape)
    y_t = np.zeros(y_train.shape)
    for i, x in enumerate(X_train):
        J[i] = grad_f(x, X_train, w, c, lm)
        fB[i] = f(x, X_train, w, c, lm)

    p = np.matmul(J.T, J)
    lhs = p + eta * np.diag(np.diag(p))
    rhs = np.dot(J.T, y_train - fB)
    inv = np.linalg.inv(lhs)
    delta = np.dot(inv, rhs)
    w_old = w
    w = w + delta
    iterations += 1
    print iterations
    print np.linalg.norm(w_old - w)
    # Convergence test
    # if np.linalg.norm(w_old - w) < eps:
    #     break

print 'u=', np.linalg.norm(u(x, X_train, w, c))
print 'prediction=', predict(X[1], X_train, w, c)

# Predict train set
print 'Evaluating model over train set...'
P_train = np.zeros(y_train.shape)
for i, x in enumerate(X_train):
    pred = predict(x, X_train, w, c)
    P_train[i] = pred
P_train[np.where(P_train > 0.5)] = 1
P_train[np.where(P_train <= 0.5)] = 0
acc = np.sum(P_train == y_train) / float(len(X_train)) * 100.
print '%.2f%s correct with n = %d\n' % (acc, '%', len(X_train))

# Predict test set
print 'Evaluating model over test set...'
P_test = np.zeros(y_test.shape)
for i, x in enumerate(X_test):
    pred = predict(x, X_train, w, c)
    P_test[i] = pred
P_test[np.where(P_test > 0.5)] = 1
P_test[np.where(P_test <= 0.5)] = 0
acc = np.sum(P_test == y_test) / float(len(X_test)) * 100.
print '%.2f%s correct with n = %d\n' % (acc, '%', len(X_test))

# Predict whole dataset
P_whole = np.zeros(y.shape)
print 'Evaluating model over the whole set...'
P_whole = np.zeros(y.shape)
for i, x in enumerate(X):
    pred = predict(x, X_train, w, c)
    P_whole[i] = pred
P_whole[np.where(P_whole > 0.5)] = 1
P_whole[np.where(P_whole <= 0.5)] = 0
acc = np.sum(P_whole == y) / float(len(X)) * 100.
print '%.2f%s correct with n = %d\n' % (acc, '%', len(X))

print 'ROC AUC:', roc_auc_score(y_test, P_test)

# Perform dim reduction
dim = X.shape[1]
if dim > 2:
    print 'Reducing dimensionality for plotting...'
    pca = PCA(2)
    X = pca.fit_transform(X)

# Create meshgrids
print 'Creating meshgrids for contour plotting...'
resolution = 500
xl = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, resolution)
yl = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, resolution)
xx, yy = np.meshgrid(xl, yl)
Z = np.c_[xx.ravel(), yy.ravel()]
U = Z

# Predict meshgrid
print 'Predicting over meshgrid...'
P_mesh = np.zeros((Z.shape[0]))
U_mesh = np.zeros((U.shape[0]))
for i, x in enumerate(Z):
    # note that we need the dataset which we trained with
    if dim > 2:
        pred = predict(pca.inverse_transform(x), X_train, w, c)
    else:
        pred = predict(x, X_train, w, c)
    P_mesh[i] = pred
# P_mesh[np.where(P_mesh > 0.5)] = 1
# P_mesh[np.where(P_mesh <= 0.5)] = 0
Z = P_mesh.reshape(xx.shape)

# Plot
print 'Plotting meshgrid...'
plt.contourf(xx, yy, Z, alpha=0.5, cmap='Spectral')
# plt.contour(xx, yy, Z)
P_mesh[np.where(P_mesh > 0.5)] = 1
P_mesh[np.where(P_mesh <= 0.5)] = 0
Z = P_mesh.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap='magma', levels=[0], corner_mask=True)

print 'Plotting scatter plot...'
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.title('Separating the plane')
plt.show()
