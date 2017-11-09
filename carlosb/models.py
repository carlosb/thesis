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
from terms import *


class MyModelLR:
    def __init__(self):
        self.weights = None
        self.inputs = None
        self.targets = None

    def predict(self, x, binary=True):
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
        prediction = sigmoid(u(x, self.inputs, self.weights, self.c))
        if binary:
            prediction = prediction > 0.5
        return prediction

    def train(self, X, y, c, lm, eta=10, it=20, display=False):
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
        self.inputs = X
        self.targets = y
        self.c = c
        self.lm = lm
        self.eta = eta

        if display:
            print 'Training model...'

        # Levenberg-Marquardt
        n = len(X)
        w = np.zeros(n)
        iterations = 0
        while iterations < it:
            # Build terms
            J = np.zeros((n, n))
            fB = np.zeros(y.shape)
            for i, x in enumerate(X):
                J[i] = grad_g(x, X, w, c, lm)
                fB[i] = g(x, X, w, c, lm)

            # Solve for delta
            p = np.matmul(J.T, J)
            lhs = p + eta * np.diag(np.diag(p))
            rhs = np.dot(J.T, y - fB)
            inv = np.linalg.inv(lhs)
            delta = np.dot(inv, rhs)

            # Update weights
            w_old = w
            w = w + delta

            if display:
                print 'Iteration: ', iterations
                print '|| w_%d - w_%d || = %f' \
                    % (iterations,
                       iterations + 1,
                       np.linalg.norm(w_old - w))

            iterations += 1
        self.weights = w


def evaluate(inputs, targets, pred, threshold=0.5):
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


def plot_decision_boundary():
    pass

