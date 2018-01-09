"""
Performs a 10-cross validation of:
- LR
- NN
- SVM
When training LR, it will perform a parameter
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

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from carlosb.models import MyModelLR
from carlosb.models import evaluate


class Benchmark:
    def __init__(self):
        self.dataset = None
        self.acc = 0
        self.roc = 0

    def __repr__(self):
        res = 'Benchmark(acc = %.6f, roc = %.6f)' % (self.acc, self.roc)
        return res

    def log_acc(self, acc):
        if acc > self.acc:
            self.acc = acc

    def log_roc(self, roc):
        if roc > self.roc:
            self.roc = roc


eta = 1.
print 'eta=', eta

# Number of folds for KFold cross validation
k = 5

# Models to benchmark
lr = MyModelLR()
nn = MLPClassifier(hidden_layer_sizes=100, alpha=0.001, activation='logistic', learning_rate='adaptive')
svm = SVC()

# Where to find the datasets
dataset_path = 'datasets/binary'
datasets = glob.glob(dataset_path + '/*.csv')
datasets.remove('datasets/binary/planning_relax.csv')

# Print which datasets where found
print 'Datasets being evaluated: '
for ds in datasets:
    path, filename = os.path.split(ds)
    print '- %s' % (filename)

# Parameter search vectors
cs = np.arange(0.1, 5, np.log(2.))
lms = np.arange(0.1, 10, 1)

# Benchmark
benchmarks = {'lr': [], 'nn': [], 'svm': []}

kf = KFold(n_splits=k, shuffle=True, random_state=42)
# Iterate over all datasets
for ds in datasets:
    # Print dataset
    print '\nBenchmarking dataset: %s' % (ds)
    # Benchmark objects
    nn_bench = Benchmark()
    svm_bench = Benchmark()
    lr_bench = Benchmark()

    nn_bench.dataset = ds
    svm_bench.dataset = ds
    lr_bench.dataset = ds

    # Load dataset
    df = pd.read_csv(ds, header=None)
    X = df.iloc[:, :-1].as_matrix()
    y = df.iloc[:, -1].as_matrix()
    print 'samples = %d' % (len(X))

    # Scale dataset
    X = scale(X)

    # Cross validate nn
    avg_acc_nn = 0
    avg_roc_nn = 0
    for train_index, test_index in kf.split(X):
        # Split datasets
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Train nn
        nn.fit(X, y)
        acc_nn, roc_nn, p = evaluate(X_test, y_test, lambda x: nn.predict([x]))
        avg_acc_nn += acc_nn
        avg_roc_nn += roc_nn

    avg_acc_nn /= k
    avg_roc_nn /= k
    nn_bench.log_acc(avg_acc_nn)
    nn_bench.log_roc(avg_roc_nn)

    # cross validate svm and lr
    it = 1
    for c in cs:

        svm = SVC(C=c, tol=1e-5)
        avg_acc_svm = 0
        avg_roc_svm = 0
        for train_index, test_index in kf.split(X):
            # Split datasets
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            svm.fit(X_train, y_train)

            acc_svm, roc_svm, p = evaluate(X_test, y_test, lambda x: svm.predict([x]))
            avg_acc_svm += acc_svm
            avg_roc_svm += roc_svm

        avg_acc_svm /= k
        avg_roc_svm /= k

        svm_bench.log_acc(avg_acc_svm)
        svm_bench.log_roc(avg_roc_svm)
        for lm in lms:
            sys.stdout.write('%d / %d\r' % (it, len(cs) * len(lms)))
            sys.stdout.flush()

            it += 1
            avg_acc_lr = 0
            avg_roc_lr = 0

            # Change svm parameters
            for train_index, test_index in kf.split(X):
                # Split datasets
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                # Train and evaluate
                lr.train(X_train, y_train, c=c, lm=lm, eta=eta, it=40)

                acc_lr, roc_lr, p = evaluate(X_test, y_test, lr.predict)

                # Add to averages
                avg_acc_lr += acc_lr
                avg_roc_lr += roc_lr

            # Average scores from k cross validation
            avg_acc_lr /= k
            avg_roc_lr /= k

            # Save only best scores
            lr_bench.log_acc(avg_acc_lr)
            lr_bench.log_roc(avg_roc_lr)
    print '\n'
    print 'LR: ', lr_bench
    print 'SVM: ', svm_bench
    print 'NN: ', nn_bench
    benchmarks['lr'].append(lr_bench)
    benchmarks['svm'].append(svm_bench)
    benchmarks['nn'].append(nn_bench)

print benchmarks
