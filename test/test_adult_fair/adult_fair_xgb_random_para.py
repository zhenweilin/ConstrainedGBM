''' XGBFairClassifier for acsincome fair task (binary classification) Batch testing
/* author: Authorname
'''
import sys 
sys.path.append("..") 
import xgboost as xgb
from lightgbm import fairness_metric
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import csv
import numpy as np
from random_paras import generate_random_paras_fair_xgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, GridSearch

fieldnames_fair_xgb = [
    "dataset",\
    "binary/multi",\
    "model",\
    "tau",\
    'sigma',\
    "outlier_threshold",\
    "max_iter_T",\
    "max_iter_L",\
    "scale_factor",\
    "alpha_max",\
    "fairness_metric(train)",\
    "diff(binary)/std(multi)(train)",\
    "accuracy(train)",\
    "fairness_metric(test)",\
    "diff(binary)/std(multi)(test)",\
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
datapath = '../../data/adult/'
data = pd.read_csv(datapath + "adult.data", header=None, on_bad_lines='skip')
data_test = pd.read_csv(datapath + "adult.test", header=None, on_bad_lines='skip')
csvname = "../save_csv/fair_xgb.csv"

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


for seed in range(25000, 25500):
# seed = 1000000
    version = "0.0.2"

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    (general_params, params) = generate_random_paras_fair_xgb(seed)

    csvdata = {
        "dataset": "adult",
        "binary/multi": "binary",
        "model" : None
    }
    print("**************** XGBFairClassifier Fair Task `adult fair task (binary classification)` ****************")
    model = xgb.XGBFairClassifier(**general_params)
    model.fit(X_train, y_train, sensitive_attribute = 9, **params)

    ## training dataset result
    yhat_train = model.predict(X_train)
    sensitive_attribute = X_train[:, 9]
    res = fairness_metric(yhat_train, y_train, sensitive_attribute)

    fairness_metric_train = res
    diff_std = abs(res[0] - res[1])
    acc_train = accuracy_score(y_train, yhat_train)
    print("fairness result(train):", fairness_metric_train)
    print("fairness difference(train):", diff_std)
    print("overall accuracy(train):", acc_train)

    csvdata["fairness_metric(train)"] = fairness_metric_train
    csvdata["diff(binary)/std(multi)(train)"] = diff_std
    csvdata["accuracy(train)"] = acc_train

    ## testing dataset result
    yhat = model.predict(X_test)
    sensitive_attribute = X_test[:, 9]
    res = fairness_metric(yhat, y_test, sensitive_attribute)
    fairness_metric_test = res
    diff_std = abs(res[0] - res[1])
    acc_test = accuracy_score(y_test, yhat)
    print("fairness result(test):", fairness_metric_test)
    print("fairness difference(test):", diff_std)
    print("overall accuracy(test):", acc_test)

    csvdata["fairness_metric(test)"] = fairness_metric_test
    csvdata["diff(binary)/std(multi)(test)"] = diff_std
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "XGBFairClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata["random_seed"] = seed

    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_fair_xgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params, **params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()



    print("**************** XGBClassifier Classification (baseline) Task `adult fair task (binary classification)` ****************")
    model = xgb.XGBClassifier(**general_params)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    sensitive_attribute = X_test[:, 9]
    res = fairness_metric(yhat, y_test, sensitive_attribute)
    print("fairness result:", res)
    print("fairness difference:", abs(res[0] - res[1]))
    print("overall accuracy:", accuracy_score(y_test, yhat))

    ## training dataset result
    yhat_train = model.predict(X_train)
    sensitive_attribute = X_train[:, 9]
    res = fairness_metric(yhat_train, y_train, sensitive_attribute)
    fairness_metric_train = res
    diff_std = abs(res[0] - res[1])
    acc_train = accuracy_score(y_train, yhat_train)
    print("fairness result(train):", fairness_metric_train)
    print("fairness difference(train):", diff_std)
    print("overall accuracy(train):", acc_train)

    csvdata["fairness_metric(train)"] = fairness_metric_train
    csvdata["diff(binary)/std(multi)(train)"] = diff_std
    csvdata["accuracy(train)"] = acc_train

    ## testing dataset result
    yhat = model.predict(X_test)
    sensitive_attribute = X_test[:, 9]
    res = fairness_metric(yhat, y_test, sensitive_attribute)
    fairness_metric_test = res
    diff_std = abs(res[0] - res[1])
    acc_test = accuracy_score(y_test, yhat)
    print("fairness result(test):", fairness_metric_test)
    print("fairness difference(test):", diff_std)
    print("overall accuracy(test):", acc_test)

    csvdata["fairness_metric(test)"] = fairness_metric_test
    csvdata["diff(binary)/std(multi)(test)"] = diff_std
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "XGBClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata["random_seed"] = seed

    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_fair_xgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()

    print("**************** Fairlearn-XGB(EG) Fair Task `adult fair task (binary classification)` ****************")
    version = "0.8.0"
    sensitive_attribute = X_train[:, 9]
    model = xgb.XGBClassifier(**general_params)
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(model, constraint)
    mitigator.fit(X_train, y_train, sensitive_features = sensitive_attribute)
    model = mitigator

    yhat_train = model.predict(X_train)
    fairness_metric_train = fairness_metric(yhat_train, y_train, sensitive_attribute)
    fairness_metric_train_array = fairness_metric(yhat_train, y_train, sensitive_attribute, dict_form = False)
    fairness_std = abs(fairness_metric_train_array[0] - fairness_metric_train_array[1])
    acc_train = accuracy_score(y_train, yhat_train)
    print("fairness result (train):", fairness_metric_train)
    print("fairness difference (train):", fairness_std)
    print("overall accuracy (train):", acc_train)
    csvdata["fairness_metric(train)"] = fairness_metric_train
    csvdata["diff(binary)/std(multi)(train)"] = fairness_std
    csvdata["accuracy(train)"] = acc_train


    # ## testing dataset result
    yhat = model.predict(X_test)
    sensitive_attribute_test = X_test[:, 9]
    fairness_metric_test = fairness_metric(yhat, y_test, sensitive_attribute_test)
    fairness_metric_test_array = fairness_metric(yhat, y_test, sensitive_attribute_test, dict_form = False)
    fairness_std = abs(fairness_metric_test_array[0] - fairness_metric_test_array[1])
    acc_test = accuracy_score(y_test, yhat)
    print("fairness result (test):", fairness_metric_test)
    print("fairness difference(test):", diff_std)
    print("overall accuracy (test):", acc_test)
    csvdata["fairness_metric(test)"] = fairness_metric_test
    csvdata["diff(binary)/std(multi)(test)"] = fairness_std
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "Fairlearn-XGB(EG)"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata["random_seed"] = seed
    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_fair_xgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()
