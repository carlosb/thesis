"""
Gets information about the datasets.
"""

import glob
import os
import pandas as pd

# Where to find the datasets
dataset_path = 'datasets/binary'
datasets = glob.glob(dataset_path + '/*.csv')
datasets.remove('datasets/binary/planning_relax.csv')

# Print which datasets where found
print 'Datasets found: '
for ds in datasets:
    path, filename = os.path.split(ds)
    print '- %s' % (filename)

# Print dataset info
print '\nInfo: '
for ds in datasets:
    path, filename = os.path.split(ds)
    print '\n- %s' % (filename)

    df = pd.read_csv(ds, header=None)
    n, dim = df.shape

    print '-- Num: %d' % (n)
    print '-- Dim: %d' % (dim - 1)
