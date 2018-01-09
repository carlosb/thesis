import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from carlosb.models import MyModelLR
from carlosb.models import evaluate

# Load dataset
df = pd.read_csv('datasets/binary/sonar.csv', header=None)

# Convert to numpy matrices
X = df.iloc[:, :-1].as_matrix()
y = df.iloc[:, -1].as_matrix()

# Preprocess
X = scale(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5)

# Declare and train model
lr = MyModelLR()
lr.train(X_train, y_train, c=0.5, lm=10, eta=10, it=20, display=True)

# Predict test set
print 'Evaluating model over test set...'
acc, roc, predictions = evaluate(X_test, y_test, lr.predict)

print 'Accuracy: ', acc * 100.
print 'ROC score: ', roc
