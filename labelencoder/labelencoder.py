"""
Description
-----------
Encodes the last column of the given dataset between 0 and (n_classes - 1)
by specfying a separator. If no separator is specified it will attempt to
automatically detect the separator. It will overwrite or create a new file
with the extension: .csv.encoded
Note: This is really just a wrapper around the pandas and sklearn functions

Usage
-----
python labelencoder.py --help

Requirements
------------
- pandas
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def strip_extension(filepath):
    filename, file_extension = os.path.splitext(filepath)
    return filename


sep_help = '''
Delimiter to use. If no sep is specified, the C engine cannot automatically
detect the separator, but the Python parsing engine can, meaning the latter
will be used and automatically detect the separator by Python\'s builtin
sniffer tool, csv.Sniffer. In addition, separators longer than 1 character and
different from '\s+' will be interpreted as regular expressions and will
also force the use of the Python parsing engine. Note that regex delimiters
are prone to ignoring quoted data. Regex example: '\r\t'
'''

description = '''
    Encodes the last column of the given dataset between 0 and (n_classes - 1)
    by specfying a separator. If no separator is specified it will attempt to
    automatically detect the separator. It will overwrite or create a new file
    with the extension: .csv.encoded
'''
parser = argparse.ArgumentParser(description)

parser.add_argument('filename', type=str,
                    help='Name of file to encode', nargs='+')
parser.add_argument('-s', '--sep', type=str, default=None,
                    help=sep_help)
parser.add_argument('-cs', '--chunksize', type=int, default=None,
                    help='''Number of bytes to read and write at a time.
                    You must pass the actual labels of the classes
                    using --classes.''')
parser.add_argument('-cl', '--classes', type=int, default=None,
                    help='The labels of the classes.')

args = parser.parse_args()

sep = args.sep
chunksize = args.chunksize
classes = args.classes
succesful_files = []
unsuccesful_files = []
print 'Encoding:'
for filepath in args.filename:
    if (chunksize is not None and classes is None):
        raise RuntimeError('You must specify the classes when using --chunksize')
    print '... %s' % filepath
    le = LabelEncoder()
    outfilename = strip_extension(filepath) + '.csv.encoded'
    if chunksize is None:
        try:
            df = pd.read_csv(filepath_or_buffer=filepath, sep=sep,
                             header=None, index_col=None,
                             engine='python')
            if classes is not None:
                le.fit(np.array(classes))
                df[df.columns[-1]] = df[df.columns[-1]].apply(le.transform)
            else:
                df[df.columns[-1]] = le.fit_transform(df[df.columns[-1]])
            df.to_csv(outfilename, mode='w+', sep=',',
                      header=None, index=False)
            succesful_files.append(filepath)
        except:
            unsuccesful_files.append(filepath)
            continue
    else:
        print 'Encoding by chunks not implemented yet. Sorry.'
        exit()

if len(succesful_files) == 0:
    print 'No files could be encoded.'
else:
    print '\nThese files were succesful: '
    for succesful in succesful_files:
        print '... %s' % succesful

if len(unsuccesful_files) == 0:
    print 'All files encoded!'
else:
    print '\nThese files could not be encoded: '
    for unsuccesful in unsuccesful_files:
        print '... %s' % unsuccesful
