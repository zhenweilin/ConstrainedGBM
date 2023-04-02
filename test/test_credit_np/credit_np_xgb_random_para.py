'''XGBNPClassifier for credit NeymanPearson task (binary classification) Batch testing
/* author: Authorname
'''
import sys 
sys.path.append("..") 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import neyman_pearson_metric
from random_paras import generate_random_paras_np_xgb
import time
import csv
version = "0.0.1"
fieldnames_np_xgb = [
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
    "max_leaves", \
    "max_depth",\
    "colsample_bytree",\
    "min_child_weight",\
    "reg_alpha",\
    "reg_lambda",\
    "gamma",\
    "time_stamp",\
    "random_seed",\
    "version"
]

# ## read data
datapath = "../../data/credit/"
csvname = "../save_csv/np_xgb.csv"
X = np.load(datapath + "X_credit.npy")
y = np.load(datapath + "y_credit.npy")

for seed in range(7000, 8000):
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    csvdata = {
    "dataset": "credit",
    "binary/multi": "binary",
    "model" : None
    }
    (general_params, params) = generate_random_paras_np_xgb(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

    print("**************** XGBNPClassifier Neyman Pearson Task `credit NeymanPearson task (binary)` ****************")
    expected_err = np.array([0.05]).ravel()
    model = xgb.XGBNPClassifier(**general_params)
    model.fit(X_train, y_train, expected_err = expected_err, **params)

    ## training dataset result
    yhat_train = model.predict(X_train)
    neyman_pearson_metric_train = neyman_pearson_metric(yhat_train, y_train)
    constraint_violation_train = np.maximum(neyman_pearson_metric_train[0] - expected_err[0], 0)
    acc_train = accuracy_score(y_train, yhat_train)
    print("neyman pearson result(train):", neyman_pearson_metric_train)
    print("constraint_violation_train:", constraint_violation_train)
    print("overall accuracy(train):", acc_train)
    csvdata["neyman_pearson_metric(train)"] = neyman_pearson_metric_train
    csvdata["constraint_violation(train)"] = constraint_violation_train
    csvdata["accuracy(train)"] = acc_train

    ## testing dataset result
    yhat_test = model.predict(X_test)
    neyman_pearson_metric_test = neyman_pearson_metric(yhat_test, y_test)
    constraint_violation_test = np.maximum(neyman_pearson_metric_test[0] - expected_err[0], 0)
    acc_test = accuracy_score(y_test, yhat_test)
    print("neyman pearson result(test):", neyman_pearson_metric_test)
    print("constraint_violation_test:", constraint_violation_test)
    print("overall accuracy(test):", acc_test)
    csvdata["neyman_pearson_metric(test)"] = neyman_pearson_metric_test
    csvdata["constraint_violation(test)"] = constraint_violation_test
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "XGBNPClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata['random_seed'] = seed
    csvdata['expected_err'] = expected_err

    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_np_xgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params, **params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()

    print("**************** XGBClassifier Classification (baseline) Task `credit NeymanPearson task (binary)` ****************")
    model = xgb.XGBClassifier(**general_params)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    ## training dataset result
    yhat_train = model.predict(X_train)
    neyman_pearson_metric_train = neyman_pearson_metric(yhat_train, y_train)
    constraint_violation_train = np.maximum(neyman_pearson_metric_train[0] - expected_err[0], 0)
    acc_train = accuracy_score(y_train, yhat_train)
    print("neyman pearson result(train):", neyman_pearson_metric_train)
    print("constraint_violation_train:", constraint_violation_train)
    print("overall accuracy(train):", acc_train)
    csvdata["neyman_pearson_metric(train)"] = neyman_pearson_metric_train
    csvdata["constraint_violation(train)"] = constraint_violation_train
    csvdata["accuracy(train)"] = acc_train

    ## testing dataset result
    yhat_test = model.predict(X_test)
    neyman_pearson_metric_test = neyman_pearson_metric(yhat_test, y_test)
    constraint_violation_test = np.maximum(neyman_pearson_metric_test[0] - expected_err[0], 0)
    acc_test = accuracy_score(y_test, yhat_test)
    print("neyman pearson result(test):", neyman_pearson_metric_test)
    print("constraint_violation_test:", constraint_violation_test)
    print("overall accuracy(test):", acc_test)
    csvdata["neyman_pearson_metric(test)"] = neyman_pearson_metric_test
    csvdata["constraint_violation(test)"] = constraint_violation_test
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "XGBClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata['random_seed'] = seed
    csvdata['expected_err'] = expected_err

    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_np_xgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()