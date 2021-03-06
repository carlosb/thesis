"""
Basic usage of the model interface. Showcases MyModelLR
and evaluate()
"""
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from carlosb.models import MyModelLR
from carlosb.models import evaluate


# Load dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Preprocess
X = scale(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5)

# Declare and train model
lr = MyModelLR()
lr.train(X_train, y_train, c=5, lm=0.01, eta=10, it=40, display=True)

# Predict test set
print 'Evaluating model over test set...'
acc, roc, predictions = evaluate(X_test, y_test, lr.predict)
print predictions

print 'Accuracy: ', acc * 100.
print 'ROC score: ', roc
