# Downloading
To download the library just clone the repository with

````git
git clone https://github.com/carlosb/thesis.git
````

# Usage
To use the package you should write your Python files in the same directory as the directory `carlosb/`. Or you can add `carlosb/` to your `PYTHONPATH` but I don't recommend it. To use the models please see the following example:

````python
"""first.py"""

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from carlosb.models import MyModelLR
from carlosb.models import evaluate


# Load dataset
X, y = make_moons(n_samples=400, noise=0.2, random_state=42)

# Preprocess
X = scale(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Declare and train model
lr = MyModelLR()
lr.train(X_train, y_train, c=5, lm=0.01, eta=10, it=20, display=True)

# Predict test set
print 'Evaluating model over test set...'
acc, roc, predictions = evaluate(X_test, y_test, lr.predict)

print 'Accuracy: ', acc * 100.
print 'ROC score: ', roc
````

You should get something like this output:
````
Accuracy:  95.0
ROC score:  0.947368421053
````

# About

Please do not hesitate to contact me at carlos.brito524@gmail.com if you have any questions.

# License [![GitHub license](https://img.shields.io/github/license/carlosb/thesis.svg)](https://github.com/carlosb/thesis/blob/master/LICENSE)

All the code is licensed under the GNU General Public License v3.0. If you have any issues with the License please keep them to yourself. I hate bureaucracy.
