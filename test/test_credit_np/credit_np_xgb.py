'''XGBNPClassifier for credit NeymanPearson task (binary classification)
/* author: Authorname
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import neyman_pearson_metric

# ## read data
datapath = "../../data/credit/"
X = np.load(datapath + "X_credit.npy")
y = np.load(datapath + "y_credit.npy")
print("num of y == 0:", np.sum(y == 0))
print("num of y == 1:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("**************** XGBNPClassifier Neyman Pearson Task `credit NeymanPearson task (binary)` ****************")
expected_err = np.array([0.05]).ravel()
model = xgb.XGBNPClassifier()
model.fit(X_train, y_train, expected_err = expected_err)
yhat = model.predict(X_test)
res = neyman_pearson_metric(yhat, y_test)
print("neyman pearson result:", res)
print("neyman pearson difference:", abs(res[0] - expected_err[0]))
print("overall accuracy:", accuracy_score(y_test, yhat))

print("**************** XGBClassifier Classification (baseline) Task `credit NeymanPearson task (binary)` ****************")
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
res = neyman_pearson_metric(yhat, y_test)
print("neyman pearson result:", res)
print("neyman pearson difference:", abs(res[0] - expected_err[0]))
print("overall accuracy:", accuracy_score(y_test, yhat))