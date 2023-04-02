''' XGBClassifier for iris (example) classification task
/* author: Authorname
'''

from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# read in the iris data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160,verbosity = 1)
model.fit(X_train, y_train)
print("**************** XGBClassifier Classification Task Example (multiple) ****************")
print("starting predict:")
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
print("Accuracy: {} ".format(accuracy_score(y_test, yhat)))

print(xgb.__version__)
print(model.get_params())

# ## binary 
y[y == 2] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, verbosity = 1)
model.fit(X_train, y_train)
print("**************** XGBClassifier Classification Task Example (binary) ****************")
print("starting predict:")
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
print("Accuracy: {} ".format(accuracy_score(y_test, yhat)))
print(xgb.__version__)
print(model.get_params())
