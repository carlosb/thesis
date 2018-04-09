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
from sklearn.model_selection import RepeatedKFold

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from carlosb.models import MyModelLR, MyModelBR
from sklearn.metrics import roc_auc_score, accuracy_score


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


max_iter = 1000
print 'max_iter=', max_iter

eta = 1
print 'eta=', eta

# Number of folds for KFold cross validation
k = 5
r = 25
print 'k=', k

# Models to benchmark
lr = MyModelBR()
nn = MLPClassifier(hidden_layer_sizes=100, alpha=0.001, activation='logistic', learning_rate='adaptive')
svm = SVC()

# Where to find the datasets
dataset_path = 'datasets/binary'
datasets = glob.glob(dataset_path + '/*.csv')

# Print which datasets where found
print 'Datasets being evaluated: '
for ds in datasets:
    path, filename = os.path.split(ds)
    print '- %s' % (filename)

# Parameter search vectors
cs = np.arange(0.001, 2.5, np.log10(2))
print 'Searching fitting degrees from %f to %f' % (cs[0], cs[len(cs) - 1])

lms = np.array([0.1, 1, 10])
print 'Searching penalty terms from %f to %f' % (lms[0], lms[len(lms) - 1])

# Benchmark
benchmarks = {'lr': [], 'nn': [], 'svm': []}

kf = RepeatedKFold(n_splits=k, n_repeats=r)
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
        avg_acc_nn += accuracy_score(y_test, nn.predict(X_test))
        avg_roc_nn += roc_auc_score(y_test, nn.predict_proba(X_test)[:, 1])

    avg_acc_nn /= (k * r)
    avg_roc_nn /= (k * r)
    nn_bench.log_acc(avg_acc_nn)
    nn_bench.log_roc(avg_roc_nn)

    # cross validate svm and lr
    it = 1
    for c in cs:
        for lm in lms:
            # ------- SVM BENCHMARK START
            svm = SVC(C=lm, gamma=c)
            avg_acc_svm = 0
            avg_roc_svm = 0
            for train_index, test_index in kf.split(X):
                # Split datasets
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                svm.fit(X_train, y_train)

                avg_acc_svm += accuracy_score(y_test, svm.predict(X_test))
                avg_roc_svm += roc_auc_score(y_test, svm.decision_function(X_test))

            avg_acc_svm /= (k * r)
            avg_roc_svm /= (k * r)

            svm_bench.log_acc(avg_acc_svm)
            svm_bench.log_roc(avg_roc_svm)
            # ------- SVM BENCHMARK END

            sys.stdout.write('%d / %d\r' % (it, len(cs) * len(lms)))
            sys.stdout.flush()

            it += 1

            # -------- BR BENCHMARK START
            avg_acc_lr = 0
            avg_roc_lr = 0

            for train_index, test_index in kf.split(X):
                # Split datasets
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                # Train and evaluate
                lr.train(X_train, y_train, c=c, lm=lm, eta=eta, eps=1e-5, max_iter=100, display=False)

                # Add to averages
                avg_acc_lr += accuracy_score(y_test, lr.predict(X_test, threshold=0.5))
                avg_roc_lr += roc_auc_score(y_test, lr.decision_function(X_test))

            # Average scores from k cross validation
            avg_acc_lr /= (k * r)
            avg_roc_lr /= (k * r)

            # Save only best scores
            lr_bench.log_acc(avg_acc_lr)
            lr_bench.log_roc(avg_roc_lr)

            # -------- BR BENCHMARK END

    print '\n'
    print 'LR: ', lr_bench
    print 'SVM: ', svm_bench
    print 'NN: ', nn_bench
    benchmarks['lr'].append(lr_bench)
    benchmarks['svm'].append(svm_bench)
    benchmarks['nn'].append(nn_bench)

print benchmarks
