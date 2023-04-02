''' LGBMNPClassifier for iris (example) fair task
/* author: Authorname
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import neyman_pearson_metric
import numpy as np
from sklearn.metrics import accuracy_score

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # train
model = lgb.LGBMNPClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, objective='multiclass_np', verbosity = 0, class_weight = {1:1,2:2,0:1}, expected_err = np.array([0.01, 0.02, 0.03]))
model.fit(X_train, y_train)
print("**************** LGBMNPClassifier Neyman Pearson Task Example (multiple) ****************")
print("starting predict:")
# test
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
# accuracy
print("Accuracy: {} ".format(accuracy_score(y_test, yhat)))

res = neyman_pearson_metric(yhat = yhat, y_test = y_test)
print("neyman pearson result:", res)
print(lgb.__version__)
print(model.get_params())

## binary 
y[y == 2] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training
model = lgb.LGBMNPClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, objective='binary_np', verbosity = 0, class_weight = {1:1,0:1}, expected_err = np.array([0.01]))
model.fit(X_train, y_train)
print("**************** LGBMNPClassifier Neyman Pearson Task Example (binary) ****************")
print("starting predict:")
# testing
yhat = model.predict(X_test)

print("yhat:", yhat)
print("y_test", y_test)
# accuracy
print("Accuracy: {} ".format(accuracy_score(y_test, yhat)))
res = neyman_pearson_metric(yhat = yhat, y_test = y_test)
print("neyman pearson result:", res)

print(model.get_params())