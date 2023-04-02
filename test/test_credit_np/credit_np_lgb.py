'''LGBMNPClassifier for credit NeymanPearson task (binary classification)
/* author: Authorname
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import neyman_pearson_metric


version = "0.0.1"
fieldnames_np_lgb = [
    "dataset",\
    "binary/multi",\
    "model",\
    "tau",\
    'sigma',\
    "outlier_threshold",\
    "max_iter_T",\
    "expected_err",\
    "neyman_pearson_metric(train)",\
    "constraint_violation(train)",\
    "accuracy(train)",\
    "neyman_pearson_metric(test)",\
    "constraint_violation(test)",\
    "accuracy(test)",\
    "n_estimators",\
    "learning_rate",\
    "num_leaves", \
    "max_depth",\
    "colsample_bytree",\
    "min_child_weight",\
    "reg_alpha",\
    "reg_lambda",\
    "time_stamp",\
    "random_seed",\
    "version"
]

# ## read data
datapath = "../../data/credit/"
X = np.load(datapath + "X_credit.npy")
y = np.load(datapath + "y_credit.npy")
print("num of y == 0:", np.sum(y == 0))
print("num of y == 1:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("**************** LGBMNPClassifier Neyman Pearson Task `credit NeymanPearson task (binary)` ****************")
expected_err = np.array([0.05]).ravel()
model = lgb.LGBMNPClassifier(expected_err = expected_err)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
res = neyman_pearson_metric(yhat, y_test)
print("neyman pearson result:", res)
print("neyman pearson difference:", abs(res[0] - expected_err[0]))
print("overall accuracy:", accuracy_score(y_test, yhat))

print("**************** LGBMClassifier Classification (baseline) Task `credit NeymanPearson task (binary)` ****************")
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
res = neyman_pearson_metric(yhat, y_test)
print("neyman pearson result:", res)
print("neyman pearson difference:", abs(res[0] - expected_err[0]))
print("overall accuracy:", accuracy_score(y_test, yhat))