'''LGBMFairClassifier for compas fair task (three class) Batch testing
/* author: Authorname
'''
import sys 
sys.path.append("..") 
import lightgbm as lgb
from lightgbm import fairness_metric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from random_paras import generate_random_paras_fair_lgb
import numpy as np
import pandas as pd
import time
import csv
from fairgbm import FairGBMClassifier
from fairlearn.reductions import GridSearch, DemographicParity
fieldnames_fair_lgb = [
    "dataset",\
    "binary/multi",\
    "model",\
    "tau",\
    'sigma',\
    "outlier_threshold",
    "max_iter_T",\
    "max_iter_L",\
    "alpha_max",\
    "scale_factor",\
    "fairness_metric(train)",\
    "diff(binary)/std(multi)(train)",\
    "accuracy(train)",\
    "fairness_metric(test)",\
    "diff(binary)/std(multi)(test)",\
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
datapath = "../../data/compas/"
raw_data = pd.read_csv(datapath + "compas-scores-two-years.csv")
raw_data = raw_data[raw_data['days_b_screening_arrest'] <= 30]
raw_data = raw_data[raw_data['days_b_screening_arrest'] >= -30]
raw_data = raw_data[raw_data['is_recid'] != -1]
raw_data = raw_data[raw_data['c_charge_degree'] != 'o']
raw_data = raw_data[raw_data['score_text'] != None]

data = raw_data[['c_charge_degree', 'age_cat', 'race', 'sex', 'score_text', 'priors_count.1', 'two_year_recid']].copy()
csvname = "../save_csv/fair_lgb.csv"

# ## encode data
for col in range(data.shape[1]):
    encoder = LabelEncoder()
    encoder.fit(data.iloc[:,col])
    data[data.columns[col]] = encoder.transform(data.iloc[:,col])

features = data[['c_charge_degree', 'age_cat', 'race', 'sex', 'priors_count.1', 'two_year_recid']]
label = data[['score_text']]


for seed in range(22000, 22000):
    version = "0.0.2"
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=seed)
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    csvdata = {
    "dataset": "compas",
    "binary/multi": "multi",
    "model" : None
    }
    (general_params, params) = generate_random_paras_fair_lgb(seed)

    print("**************** LGBMFairClassifier Fair Task `compas fair task (three class)` ****************")
    model = lgb.LGBMFairClassifier(**{**general_params, **params})
    sensitive_attribute = X_train['race'].to_numpy()
    model.fit(X_train, y_train.values.ravel(), sensitive_attribute = sensitive_attribute)

    # ## training dataset result
    yhat_train = model.predict(X_train)
    sensitive_attribute = X_train['race'].to_numpy()
    fairness_metric_train = fairness_metric(yhat_train, y_train, sensitive_attribute)
    fairness_metric_train_array = fairness_metric(yhat_train, y_train, sensitive_attribute, dict_form = False)
    fairness_std = np.std(fairness_metric_train_array)
    acc_train = accuracy_score(y_train, yhat_train)
    print("fairness result (train):", fairness_metric_train)
    print("fairness standard deviation (train):", fairness_std)
    print("overall accuracy (train):", acc_train)

    csvdata["fairness_metric(train)"] = fairness_metric_train
    csvdata["diff(binary)/std(multi)(train)"] = fairness_std
    csvdata["accuracy(train)"] = acc_train

    # ## testing dataset result
    yhat = model.predict(X_test)
    sensitive_attribute_test = X_test['race'].to_numpy()
    fairness_metric_test = fairness_metric(yhat, y_test, sensitive_attribute_test)
    fairness_metric_test_array = fairness_metric(yhat, y_test, sensitive_attribute_test, dict_form = False)
    fairness_std = np.std(fairness_metric_test_array)
    acc_test = accuracy_score(y_test, yhat)
    print("fairness result (test):", fairness_metric_test)
    print("fairness standard deviation (test):", fairness_std)
    print("overall accuracy (test):", acc_test)
    csvdata["fairness_metric(test)"] = fairness_metric_test
    csvdata["diff(binary)/std(multi)(test)"] = fairness_std
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "LGBMFairClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata["random_seed"] = seed
    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_fair_lgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params, **params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()


    print("**************** LGBMClassifier Classification (baseline) Task `adult fair task (binary classification)` ****************")
    model = lgb.LGBMClassifier(**general_params)
    model.fit(X_train, y_train.values.ravel())
    # ## training dataset result
    yhat_train = model.predict(X_train)
    sensitive_attribute = X_train['race'].to_numpy()
    fairness_metric_train = fairness_metric(yhat_train, y_train, sensitive_attribute)
    fairness_metric_train_array = fairness_metric(yhat_train, y_train, sensitive_attribute, dict_form = False)
    fairness_std = np.std(fairness_metric_train_array)
    acc_train = accuracy_score(y_train, yhat_train)
    print("fairness result (train):", fairness_metric_train)
    print("fairness standard deviation (train):", fairness_std)
    print("overall accuracy (train):", acc_train)
    csvdata["fairness_metric(train)"] = fairness_metric_train
    csvdata["diff(binary)/std(multi)(train)"] = fairness_std
    csvdata["accuracy(train)"] = acc_train

    # ## testing dataset result
    yhat = model.predict(X_test)
    sensitive_attribute_test = X_test['race'].to_numpy()
    fairness_metric_test = fairness_metric(yhat, y_test, sensitive_attribute_test)
    fairness_metric_test_array = fairness_metric(yhat, y_test, sensitive_attribute_test, dict_form = False)
    fairness_std = np.std(fairness_metric_test_array)
    acc_test = accuracy_score(y_test, yhat)
    print("fairness result (test):", fairness_metric_test)
    print("fairness standard deviation (test):", fairness_std)
    print("overall accuracy (test):", acc_test)
    csvdata["fairness_metric(test)"] = fairness_metric_test
    csvdata["diff(binary)/std(multi)(test)"] = fairness_std
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "LGBMClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata["random_seed"] = seed
    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_fair_lgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()

    print("**************** FairGBMClassifier Fair Task `adult fair task (binary classification)` ****************")
    version = "0.9.14"
    sensitive_attribute = X_train['race'].to_numpy()
    model = FairGBMClassifier(**general_params, multiplier_learning_rate = params['sigma'], constraint_type="FNR")
    model.fit(X_train, y_train, constraint_group=sensitive_attribute)
    ## trainging dataset result
    yhat_train = model.predict(X_train)
    fairness_metric_train = fairness_metric(yhat_train, y_train, sensitive_attribute)
    fairness_metric_train_array = fairness_metric(yhat_train, y_train, sensitive_attribute, dict_form = False)
    fairness_std = np.std(fairness_metric_train_array)
    acc_train = accuracy_score(y_train, yhat_train)
    print("fairness result (train):", fairness_metric_train)
    print("fairness standard deviation (train):", fairness_std)
    print("overall accuracy (train):", acc_train)

    # ## testing dataset result
    yhat = model.predict(X_test)
    sensitive_attribute_test = X_test['race'].to_numpy()
    fairness_metric_test = fairness_metric(yhat, y_test, sensitive_attribute_test)
    fairness_metric_test_array = fairness_metric(yhat, y_test, sensitive_attribute_test, dict_form = False)
    fairness_std = np.std(fairness_metric_test_array)
    acc_test = accuracy_score(y_test, yhat)
    print("fairness result (test):", fairness_metric_test)
    print("fairness standard deviation (test):", fairness_std)
    print("overall accuracy (test):", acc_test)
    csvdata["fairness_metric(test)"] = fairness_metric_test
    csvdata["diff(binary)/std(multi)(test)"] = fairness_std
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "FairGBMClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata["random_seed"] = seed
    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_fair_lgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params, "sigma": params['sigma']}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()
