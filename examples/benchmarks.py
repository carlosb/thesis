"""
Performs a 10-cross validation of the my models.
When training the models, it will perform a parameter
for c in (0, 5) and lambda in (0, 10). Eta is fixed to 10.

The program will output the current dataset being used for
benchmarking while also outputing the scores for each dataset.
When a dataset has finished being benchmarked it will output the following:

Benchmark(c, lm, acc, roc)

Each of these entries will be the best scores for the particular choices of
c and lm.

"""
import sys
import glob
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.model_selection import KFold

from carlosb.models import MyModelLR
from carlosb.models import evaluate


class Benchmark:
    def __init__(self):
        self.dataset = None
        self.c = 0
        self.lm = 0
        self.acc = 0
        self.roc = 0

    def __repr__(self):
        res = 'Benchmark(c = %.6f, lm = %.6f, acc = %.6f, roc = %.6f)' % (
            self.c, self.lm, self.acc, self.roc)
        return res


# Number of folds for KFold cross validation
k = 5

# Models to benchmark
lr = MyModelLR()

# Where to find the datasets
dataset_path = 'datasets/binary'
datasets = glob.glob(dataset_path + '/*.csv')

# Print which datasets where found
print 'Datasets being evaluated: '
for ds in datasets:
    path, filename = os.path.split(ds)
    print '- %s' % (filename)

# Parameter search vectors
cs = np.arange(0.1, 5, np.log(2.))
lms = np.arange(0, 10, 2)

# Benchmark list
benchmarks = []

# Iterate over all datasets
for ds in datasets:
    # Print dataset
    print '\nBenchmarking dataset: %s' % (ds)
    bench = Benchmark()
    bench.dataset = ds

    # Load dataset
    df = pd.read_csv(ds, header=None)
    X = df.iloc[:, :-1].as_matrix()
    y = df.iloc[:, -1].as_matrix()
    print 'samples = %d' % (len(X))

    # Scale dataset
    X = scale(X)

    # Cross validate model
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    current_iterations = 1
    total_iterations = len(cs) * len(lms)
    for c in cs:
        for lm in lms:
            ki = 1
            avg_acc = 0.
            avg_roc = 0.
            for train_index, test_index in kf.split(X):
                # Split datasets
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                # Train and evaluate
                lr.train(X_train, y_train, c=c, lm=lm, eta=0.1, it=20)
                acc, roc, predictions = evaluate(X_test, y_test, lr.predict)

                # Add to averages
                avg_acc += acc
                avg_roc += roc

                # Print progress
                sys.stdout.write('c = %.4f | lm = %.4f | it = %d / %d | k = %d / %d | acc = %.4f | roc = %.4f\r' %
                                 (c, lm, current_iterations, total_iterations, ki, k, acc, roc))
                sys.stdout.flush()

                ki += 1

            # Average scores from k cross validation
            avg_acc /= k
            avg_roc /= k

            # Save only best scores
            if avg_acc > bench.acc:
                bench.c = c
                bench.lm = lm
                bench.acc = avg_acc

            if avg_roc > bench.roc:
                bench.roc = avg_roc

            # Increment iterations
            current_iterations += 1
    print '\n'
    print bench
    benchmarks.append(bench)

print benchmarks
