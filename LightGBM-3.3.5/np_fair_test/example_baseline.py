''' LGBMClassifier for iris (example) classification task
/* author: Authorname
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# train
model = lgb.LGBMClassifier(max_depth=4, learning_rate=0.1, n_estimators=160, verbosity = 0)
model.fit(X_train, y_train)
print("**************** LGBMClassifier Classification Task Example (multiple) ****************")
print("starting predict:")
# test
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
# accuracy
print("Accuracy: {} ".format(accuracy_score(y_test, yhat)))

print(lgb.__version__)
print(model.get_params())

# ## binary 
y[y == 2] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training
model = lgb.LGBMClassifier(max_depth=4, learning_rate=0.1, n_estimators=160, verbosity = 0)
model.fit(X_train, y_train)
print("**************** LGBMClassifier Classification Task Example (binary) ****************")
print("starting predict:")
# testing
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
# accuracy
print("Accuracy: {} ".format(accuracy_score(y_test, yhat)))

print(model.get_params())
