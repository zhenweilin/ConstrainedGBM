'''XGBFairClassifier for compas fair task (three classes)
/* author: Authorname
'''

import xgboost as xgb
from lightgbm import fairness_metric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# ## read data
datapath = "../../data/compas/"
raw_data = pd.read_csv(datapath + "compas-scores-two-years.csv")
raw_data = raw_data[raw_data['days_b_screening_arrest'] <= 30]
raw_data = raw_data[raw_data['days_b_screening_arrest'] >= -30]
raw_data = raw_data[raw_data['is_recid'] != -1]
raw_data = raw_data[raw_data['c_charge_degree'] != 'o']
raw_data = raw_data[raw_data['score_text'] != None]

data = raw_data[['c_charge_degree', 'age_cat', 'race', 'sex', 'score_text', 'priors_count.1', 'two_year_recid']].copy()

# ## encode data
for col in range(data.shape[1]):
    encoder = LabelEncoder()
    encoder.fit(data.iloc[:,col])
    data[data.columns[col]] = encoder.transform(data.iloc[:,col])

features = data[['c_charge_degree', 'age_cat', 'race', 'sex', 'priors_count.1', 'two_year_recid']]
label = data[['score_text']]

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)

# ## xgb parameter
general_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_leaves": 31,
    "max_depth": 8,
    "gamma": 0.5,
    "min_child_weight": 1,
    "reg_alpha":0.2,
    "reg_lambda":0.2,
}
params = {
    "tau" : np.array([0.1]),
    "sigma" : np.array([0.1]),
    "outlier_threshold" : np.array([10]),
    "max_iter_T": 50,
    "max_iter_L": 20,
}

print("**************** XGBFairClassifier Fair Task `compas fair task (three class)` ****************")
model = xgb.XGBFairClassifier(**general_params)
sensitive_attribute_train = X_train['race'].to_numpy()
model.fit(X_train, y_train.values.ravel(), sensitive_attribute = sensitive_attribute_train, **params)
yhat = model.predict(X_test)
sensitive_attribute_test = X_test['race'].to_numpy()
res = fairness_metric(yhat, y_test, sensitive_attribute_test)
print("fairness result:", res)
res_array = fairness_metric(yhat, y_test, sensitive_attribute_test, dict_form = False)
print("fairness variance:", np.std(res_array))
print("overall accuracy:", accuracy_score(y_test, yhat))

print("**************** XGBClassifier Classification (baseline) Task `compas fair task (three class)` ****************")
model = xgb.XGBClassifier(**general_params)
model.fit(X_train, y_train.values.ravel())
yhat = model.predict(X_test)
sensitive_attribute_test = X_test['race'].to_numpy()
res = fairness_metric(yhat, y_test, sensitive_attribute_test)
print("fairness result:", res)
res_array = fairness_metric(yhat, y_test, sensitive_attribute_test, dict_form = False)
print("fairness variance:", np.std(res_array))
print("overall accuracy:", accuracy_score(y_test, yhat))