''' XGBNPClassifier for iris (example) fair task
/* author: Authorname
'''
from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = xgb.XGBNPClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, objective='multi:softprob_np', verbosity = 1)
model.fit(X_train, y_train, expected_err = np.array([0.05, 0.01, 0.04]))
print("**************** XGBNPClassifier Neyman Pearson Task Example (multiple) ****************")
print("starting predict:")
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
print("Accuracy: {}".format(accuracy_score(y_test, yhat)))

print(xgb.__version__)
print(model.get_params())

## binary 
y[y == 2] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = xgb.XGBNPClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, objective='binary:logistic_np', verbosity = 1)
model.fit(X_train, y_train, expected_err = np.array([0.03]))
print("**************** XGBNPClassifier Neyman Pearson Task Example (binary) ****************")
print("starting predict:")
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
print("Accuracy: {}".format(accuracy_score(y_test, yhat)))

print(xgb.__version__)
print(model.get_params())