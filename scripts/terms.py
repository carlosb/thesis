"""terms.py

This file contains the useful mathematical expressions
that are used in the training stage of our models.
"""

import numpy as np


def sigmoid(x):
    """Numerically-stable sigmoid function.

    x : array_like
        Input.
    """
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def sigmoid_derivative(x):
    """Derivative of sigmoid
    """
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


def laplacian_phi(x, ci, c):
    """Laplacian of RBF
    x : array_like
        Input.
    ci : array_like
        Center of RBF kernel.
    c : float
        Constant parameter.
    """
    return c * (c * np.linalg.norm(x - ci, axis=1)**2 - len(x)) * phi(x, ci, c)


def u(x, cs, w, c):
    """Radial Basis Function Approximation.

    Parameters
    ----------
    x : array_like
        Points to predict
    w : array_like
        Weight matrix.
    cs : array_like
        Center matrix of RBFs
    c : float
        Parameter constant
    """
    return np.dot(w, phi(x, cs, c))


def laplacian_u(x, cs, w, c):
    """Laplacian of u.

    Parameters
    ----------
    x : array_like
        Input.
    cs : array_like
        Center matrix of the RBF kernels.
    w : array_like
        Weights matrix.
    c : float
        Constant parameter.
    """
    return np.dot(w, laplacian_phi(x, cs, c))


def gradw_sigmoid(x, cs, w, c):
    """Gradient with respect to the weights of the sigmoid

    Parameters
    ----------
    x : array_like
        Input.
    cs : array_like
        Center matrix of the RBF kernels.
    w : array_like
        Weights matrix.
    c : float
        Constant parameter.
    """
    return sigmoid_derivative(u(x, cs, w, c)) * phi(x, cs, c)


def gradw_laplacian_u(x, cs, c):
    """Gradient with respect to the weights of the laplacian of u.

    Parameters
    ----------
    x : array_like
        Input.
    cs : array_like
        Center matrix of the RBF kernels.
    c : float
        Constant parameter.
    """
    return laplacian_phi(x, cs, c)


def g(x, cs, w, c, lm):
    return sigmoid(u(x, cs, w, c)) - lm * laplacian_u(x, cs, w, c)


def grad_g(x, cs, w, c, lm):
    return gradw_sigmoid(x, cs, w, c) - lm * gradw_laplacian_u(x, cs, c)


def predict(x, cs, w, c):
    return sigmoid(u(x, cs, w, c))
