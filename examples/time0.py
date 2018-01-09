import time
import glob
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from carlosb.models import MyModelLR

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

for ds in datasets:
    print 'Current dataset: %s' % (ds)

    # Load dataset
    df = pd.read_csv(ds, header=None)
    X = df.iloc[:, :-1].as_matrix()
    y = df.iloc[:, -1].as_matrix()
    print 'N = %d' % (X.shape[0])
    print 'dim = %d' % (X.shape[1])

    # Scale dataset
    X = scale(X)

    # Measure time
    t1 = time.clock()
    lr.train(X, y, 0.001, 0.1)
    t2 = time.clock()
    print 'lr: %f' % (t2 - t1)

    t1 = time.clock()
    nn.fit(X, y)
    t2 = time.clock()
    print 'nn: %f' % (t2 - t1)

    t1 = time.clock()
    svm.fit(X, y)
    t2 = time.clock()
    print 'svm: %f' % (t2 - t1)
