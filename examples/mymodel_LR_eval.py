"""
This file contains the code necessary to evaluate
the model described in Chapter 5 "A New Variational Model for Supervised
Learning" from my thesis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_moons
from models import MyModelLR


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
    Returns a tuple (accuracy, roc, predictions)

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
        predictions[i] = pred(x) > threshold
    acc = np.sum(predictions == targets) / float(len(inputs))
    roc = roc_auc_score(targets, predictions)
    return acc, roc, predictions


X, y = make_moons(n_samples=400, noise=0.3)

# Preprocess
X = scale(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16)

# Declare and train model
lr = MyModelLR()
lr.train(X_train, y_train, c=3, lm=0.01, eta=10, it=20)

# Predict train set
print 'Evaluating model over train set...'
acc, roc, predictions = evaluate(X_train, y_train, lr.predict)
print acc

# Predict train set
print 'Evaluating model over test set...'
acc, roc, predictions = evaluate(X_test, y_test, lr.predict)
print acc

