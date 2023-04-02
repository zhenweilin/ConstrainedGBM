''' LGBMFairClassifier for adult fair task (binary classification)
/* author: Authorname
'''

import lightgbm as lgb
from lightgbm import fairness_metric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# ## read data
datapath = '../../data/adult/'
data = pd.read_csv(datapath + "adult.data", header=None, on_bad_lines='skip')
data_test = pd.read_csv(datapath + "adult.test", header=None, on_bad_lines='skip')


## label encoder column
column_list = [1, 3, 5, 6, 7, 8, 9, 13, 14]
for col in column_list:
    encoder = LabelEncoder()
    encoder.fit(data.iloc[:,col])
    data[data.columns[col]] = encoder.transform(data.iloc[:,col])
    data_test[data_test.columns[col]] = encoder.transform(data_test.iloc[:,col])
    
X_train = data.iloc[:,0:-1].to_numpy()
y_train = data.iloc[:,-1].to_numpy()

X_test = data_test.iloc[:,0:-1].to_numpy()
y_test = data_test.iloc[:,-1].to_numpy()


# ## lgbm parameter
params = {
    "tau" : np.array([0.1]),
    "sigma" : np.array([0.1]),
    "outlier_threshold" : np.array([10]),
    "max_iter_T": 50,
    "max_iter_L": 30,
    "scale_factor": np.array([1])
}

print("**************** LGBMFairClassifier Fair Task `adult fair task (binary classification)` ****************")
model = lgb.LGBMFairClassifier(**params)
model.fit(X_train, y_train, sensitive_attribute = 9)
yhat = model.predict(X_test)
sensitive_attribute = X_test[:, 9]
res = fairness_metric(yhat, y_test, sensitive_attribute)
print("fairness result:", res)
print("fairness difference:", abs(res[0] - res[1]))
print("overall accuracy:", accuracy_score(y_test, yhat))

print("**************** LGBMClassifier Classification (baseline) Task `adult fair task (binary classification)` ****************")
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
sensitive_attribute = X_test[:, 9]
res = fairness_metric(yhat, y_test, sensitive_attribute)
print("fairness result:", res)
print("fairness difference:", abs(res[0] - res[1]))
print("overall accuracy:", accuracy_score(y_test, yhat))
