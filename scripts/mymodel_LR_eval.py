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
from models import MyModelLR


# Dictionary to convert to numeric data
target_cleanup = {1: 1, 2: 0}

# Load dataset
df = pd.read_csv('../datasets/planning_relax/plrx.txt', sep='\s+', header=None)

# Remove target column from inputs and extract it
inputs = df.drop([len(df.columns) - 1], axis=1)
targets = df[len(df.columns) - 1].replace(target_cleanup)

# Convert to matrix
X = inputs.as_matrix()
y = targets.as_matrix()

# Preprocess
X = scale(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

lr = MyModelLR()
lr.train(X_train, y_train, c=5, lm=0.01, eta=10, it=20)

# Predict train set
print 'Evaluating model over train set...'
P_train = np.zeros(y_train.shape)
for i, x in enumerate(X_train):
    pred = lr.predict(x)
    P_train[i] = pred
P_train[np.where(P_train > 0.5)] = 1
P_train[np.where(P_train <= 0.5)] = 0
acc = np.sum(P_train == y_train) / float(len(X_train)) * 100.
print '%.2f%s correct with n = %d\n' % (acc, '%', len(X_train))

# Predict test set
print 'Evaluating model over test set...'
P_test = np.zeros(y_test.shape)
for i, x in enumerate(X_test):
    pred = lr.predict(x)
    P_test[i] = pred
P_test[np.where(P_test > 0.5)] = 1
P_test[np.where(P_test <= 0.5)] = 0
acc = np.sum(P_test == y_test) / float(len(X_test)) * 100.
print '%.2f%s correct with n = %d\n' % (acc, '%', len(X_test))

print 'ROC AUC:', roc_auc_score(y_test, P_test)
