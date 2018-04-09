"""
models.py

This file contains our variational learning models. All of the models
have *at least* the following methods:

````python
model.predict(x, binary=True)
model.train(X, y, ...)

Copyright (C) Carlos David Brito Pacheco (carlos.brito524@gmail.com)
````
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from terms import *


class MyModelLR:
    def __init__(self):
        self.weights = None
        self.inputs = None
        self.targets = None

    def predict(self, x, threshold=None):
        """Predicts one input.

        Parameters
        ----------
        x : array_like vector
            An n-dimensional vector that represents an input to predict.
        binary : bool
            True to return either 1 or 0. False to return a value
            in the open interval (0, 1)
        Returns
        -------

        """
        s = sigmoid(u(x, self.inputs, self.weights, self.c))
        lu = laplacian_u(x, self.inputs, self.weights, self.c, self.d)
        prediction = s - self.lm * lu
        if threshold is not None:
            prediction = int(prediction >= threshold)
        return prediction

    def predict_proba(self, x):
        return sigmoid(u(x, self.inputs, self.weights, self.c))

    def train(self, X, y, c, lm, eta=10, eps=1e-5, max_iter=200, display=False):
        """Trains the model.

        Parameters
        ----------
        X   : array_like matrix
            Matrix of inputs.
        y   : array_like vector
            Column vector of targets
        c   : float (> 0)
            Fitting degree. Lower is more linear. Very high values may cause
            overfitting.
        lm  : float (> 0)
            Regularization parameter.
        eta : float (> 0)
            Dampening parameter.
        it  : integer (> 0)
            Number of iterations to run for.
        """
        n = X.shape[0]  # number of samples
        d = X.shape[1]  # number of dimensions

        if c == 'auto':
            c = 1.0 / d

        self.inputs = X
        self.targets = y
        self.c = c
        self.lm = lm
        self.eta = eta
        self.d = X.shape[1]

        if display:
            print 'Training model...'

        # Levenberg-Marquardt
        w = np.zeros(n)
        iterations = 0
        while iterations < max_iter:
            # Build terms
            J = np.zeros((n, n))
            fB = np.zeros(y.shape)
            for i, x in enumerate(X):
                J[i] = grad_g(x, X, w, c, lm, d)
                fB[i] = g(x, X, w, c, lm, d)

            # Solve for delta
            p = np.matmul(J.T, J)
            lhs = p + eta * np.diag(np.diag(p))
            rhs = np.dot(J.T, y - fB)
            inv = np.linalg.inv(lhs)
            delta = np.dot(inv, rhs)

            # Update weights
            w_old = w
            w = w + delta

            if np.linalg.norm(w_old - w) < eps:
                break

            if display:
                print 'Iteration: ', iterations
                print '|| w_%d - w_%d || = %f' \
                    % (iterations,
                       iterations + 1,
                       np.linalg.norm(w_old - w))

            iterations += 1
        self.weights = w

    def complexity_generated(self, x):
        lu = laplacian_u(x, self.inputs, self.weights, self.c, self.d)
        return -lu

    def complexity(self, x):
        nu = np.linalg.norm(grad_u(x, self.inputs, self.weights, self.c))
        return 0.5 * nu**2

    def complexity_flow(self, x):
        gu = grad_u(x, self.inputs, self.weights, self.c)
        return -gu


class MyModelBR:
    def __init__(self):
        self.weights = None
        self.inputs = None
        self.targets = None

    def predict(self, x, threshold=None):
        """Predicts one input.

        Parameters
        ----------
        x : array_like vector
            An n-dimensional vector that represents an input to predict.
        binary : bool
            True to return either 1 or 0. False to return a value
            in the open interval (0, 1)
        Returns
        -------

        """
        if x.ndim is 1:
            s = sigmoid(u(x, self.inputs, self.weights, self.c))
            bu = biharmonic_u(x, self.inputs, self.weights, self.c, self.d)
            prediction = s + self.lm * bu
            if threshold is not None:
                prediction = int(prediction >= threshold)
            return prediction
        elif x.ndim is 2:
            prediction = np.zeros(x.shape[0])
            for i, xi in enumerate(x):
                s = sigmoid(u(xi, self.inputs, self.weights, self.c))
                bu = biharmonic_u(xi, self.inputs, self.weights, self.c, self.d)
                prediction[i] = s + self.lm * bu
                if threshold is not None:
                    prediction[i] = int(prediction[i] >= threshold)
            return prediction
        else:
            raise ValueError('x must be a numpy array of dimension <= 2')

    def decision_function(self, x):
        return self.predict(x, threshold=None)

    def train(self, X, y, c='auto', lm=1.0, eta=10, eps=1e-3, max_iter=-1, display=False):
        """Trains the model.

        Parameters
        ----------
        X   : array_like matrix
            Matrix of inputs.
        y   : array_like vector
            Column vector of targets
        c   : float (> 0)
            Fitting degree. Lower is more linear. Very high values may cause
            overfitting.
        lm  : float (> 0)
            Regularization parameter.
        eta : float (> 0)
            Dampening parameter.
        it  : integer (> 0)
            Number of iterations to run for.
        """
        n = X.shape[0]  # number of samples
        d = X.shape[1]  # number of dimensions

        if c == 'auto':
            c = 1.0 / d

        self.inputs = X
        self.targets = y
        self.c = c
        self.lm = lm
        self.eta = eta
        self.d = d

        if display:
            print 'Training model...'

        # Levenberg-Marquardt
        w = np.zeros(n)
        iterations = 0
        while (iterations < max_iter) or (max_iter == -1):
            # Build terms
            J = np.zeros((n, n))
            fB = np.zeros(y.shape)
            for i, x in enumerate(X):
                J[i] = grad_g_br(x, X, w, c, lm, d)
                fB[i] = g_br(x, X, w, c, lm, d)

            # Solve for delta
            p = np.matmul(J.T, J)
            lhs = p + eta * np.diag(np.diag(p))
            rhs = np.dot(J.T, y - fB)
            inv = np.linalg.inv(lhs)
            delta = np.dot(rhs, inv)

            # Update weights
            w_old = w
            w = w + delta

            if np.linalg.norm(w_old - w) < eps:
                break

            if display:
                print 'Iteration: ', iterations
                print '|| d_%d || = %f' \
                    % (iterations,
                       np.linalg.norm(delta))

            iterations += 1
        self.weights = w

    def complexity_generated(self, x):
        bu = biharmonic_u(x, self.inputs, self.weights, self.c, self.d)
        return bu

    def complexity(self, x):
        lu = laplacian_u(x, self.inputs, self.weights, self.c, self.d)
        # lu = biharmonic_u(x, self.inputs, self.weights, self.c, self.d)
        return lu**2

    def complexity_flow(self, x):
        t = np.zeros(x.shape)
        for i, ci in enumerate(self.inputs):
            const = self.c**2 * (2 + self.d - self.c *
                                 np.linalg.norm(x - ci)**2)
            t += const * self.weights[i] * (x - ci) * phi(x, [ci], self.c)
        return t


def evaluate(inputs, targets, pred, threshold=None):
    """ Evaluates a prediction rule over a dataset by
    calculating the total accuracy and the ROC score.

    Parameters
    ----------
    inputs : array_like
        Matrix of inputs.
    targets : array_like
        Target vector.
    pred : function
        The function that will be used to predict

    Returns
    -------
    Returns a tuple (acc, roc, predictions)

    acc : float
        The accuracy of the predictions over the given inputs
        in the range 0 to 1.
    roc : float
        The ROC score over the inputs.
    predictions : array_like
        A prediction vector which contains all the predictions.
    """
    predictions = np.zeros(targets.shape)
    for i, x in enumerate(inputs):
        predictions[i] = pred(x)

    if threshold is not None:
        predictions = predictions > threshold

    acc = np.sum(predictions == targets) / float(len(inputs))
    roc = roc_auc_score(targets, predictions)
    return acc, roc, predictions
