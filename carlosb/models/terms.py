"""
terms.py

This file contains the useful mathematical expressions
that are used in the training stage of our models.

Copyright (C) Carlos David Brito Pacheco (carlos.brito524@gmail.com)
"""

import numpy as np


def sigmoid(x):
    """Sigmoid function.

    x : array_like
        Input.
    """
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


def laplacian_phi(x, ci, c, d):
    """Laplacian of RBF
    x : array_like
        Input.
    ci : array_like
        Center of RBF kernel.
    c : float
        Constant parameter.
    """
    return c * (c * np.linalg.norm(x - ci, axis=1)**2 - d) * phi(x, ci, c)


def biharmonic_phi(x, xi, c, d):
    """Returns biharmonic of phi
    """
    norm = np.linalg.norm(x - xi, axis=1)
    ph = phi(x, xi, c)
    f1 = c**2 * d * (2 + d - c * norm**2) * ph
    f2 = c**3 * norm**2 * (4 + d - c * norm**2) * ph
    return f1 - f2


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


def grad_u(x, cs, w, c):
    t = np.zeros(x.shape)
    for i, ci in enumerate(cs):
        t += w[i] * (x - ci) * phi(x, [ci], c)
    return -c * t


def laplacian_u(x, cs, w, c, d):
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
    return np.dot(w, laplacian_phi(x, cs, c, d))


def biharmonic_u(x, cs, w, c, d):
    return np.dot(w, biharmonic_phi(x, cs, c, d))


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


def gradw_laplacian_u(x, cs, c, d):
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
    return laplacian_phi(x, cs, c, d)


def gradw_biharmonic_u(x, cs, c, d):
    return biharmonic_phi(x, cs, c, d)


def g(x, cs, w, c, lm, d):
    return sigmoid(u(x, cs, w, c)) - lm * laplacian_u(x, cs, w, c, d)


def grad_g(x, cs, w, c, lm, d):
    return gradw_sigmoid(x, cs, w, c) - lm * gradw_laplacian_u(x, cs, c, d)


def g_br(x, cs, w, c, lm, d):
    if lm == 0:
        return sigmoid(u(x, cs, w, c))
    else:
        return sigmoid(u(x, cs, w, c)) + lm * biharmonic_u(x, cs, w, c, d)


def grad_g_br(x, cs, w, c, lm, d):
    if lm == 0:
        return gradw_sigmoid(x, cs, w, c)
    else:
        return gradw_sigmoid(x, cs, w, c) + lm * gradw_biharmonic_u(x, cs, c, d)


def predict(x, cs, w, c):
    return sigmoid(u(x, cs, w, c))
