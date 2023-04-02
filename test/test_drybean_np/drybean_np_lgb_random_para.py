'''LGBMNPClassifier for drybean NeymanPearson task (7-classes) Batch testing
/* author: Authorname 
'''
import sys 
sys.path.append("..") 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import neyman_pearson_metric
from random_paras import generate_random_paras_np_lgb
import csv
import time

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
csvname = "../save_csv/np_lgb.csv"
datapath = "../../data/DryBeanDataset/"
data = pd.read_excel(datapath + "Dry_Bean_Dataset.xlsx")
X = data.iloc[:, 0:-2]
y = data['Class']
X = X.to_numpy(dtype=np.float32)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y.values)
X = StandardScaler().fit(X).transform(X)

for seed in range(1000, 2000):
    version = "0.0.4"
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    csvdata = {
    "dataset": "drybean",
    "binary/multi": "multi",
    "model" : None
    }
    (general_params, params) = generate_random_paras_np_lgb(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

    print("**************** LGBMNPClassifier Neyman Pearson Task `drybean NeymanPearson task (7-classes)` ****************")
    eps = 1e-8
    expected_err = np.array([1-eps, 0.01, 0.03, 0.02, 0.02, 1-eps, 1-eps])
    expected_err = expected_err.ravel()
    model = lgb.LGBMNPClassifier(**{**general_params, **params}, expected_err = expected_err)
    model.fit(X_train, y_train)

    ## training dataset result
    yhat_train = model.predict(X_train)
    neyman_pearson_metric_train = neyman_pearson_metric(yhat_train, y_train)
    neyman_pearson_metric_train_array = neyman_pearson_metric(yhat_train, y_train, dict_form = False)
    constraint_violation_train = np.sum(np.maximum( neyman_pearson_metric_train_array - expected_err, 0))
    acc_train = accuracy_score(y_train, yhat_train)
    print("neyman pearson result(train):", neyman_pearson_metric_train)
    print("constraint_violation_train:", constraint_violation_train)
    print("overall accuracy(train):", acc_train)
    csvdata["neyman_pearson_metric(train)"] = neyman_pearson_metric_train
    csvdata["constraint_violation(train)"] = constraint_violation_train
    csvdata["accuracy(train)"] = acc_train
    csvdata['expected_err'] = None

    ## testing dataset result
    yhat_test = model.predict(X_test)
    neyman_pearson_metric_test = neyman_pearson_metric(yhat_test, y_test)
    neyman_pearson_metric_test_array = neyman_pearson_metric(yhat_test, y_test, dict_form = False)
    constraint_violation_test = np.sum(np.maximum( neyman_pearson_metric_test_array - expected_err, 0))
    acc_test = accuracy_score(y_test, yhat_test)
    print("neyman pearson result(test):", neyman_pearson_metric_test)
    print("constraint_violation_test:", constraint_violation_test)
    print("overall accuracy(test):", acc_test)
    csvdata["neyman_pearson_metric(test)"] = neyman_pearson_metric_test
    csvdata["constraint_violation(test)"] = constraint_violation_test
    csvdata["accuracy(test)"] = acc_test

    ## model info
    csvdata["model"] = "LGBMNPClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata['random_seed'] = seed
    csvdata['expected_err'] = None

    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_np_lgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params, **params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()

    print("**************** LGBMClassifier Classification (baseline) Task `drybean NeymanPearson task (7-classes)` ****************")
    model = lgb.LGBMClassifier(**general_params)
    model.fit(X_train, y_train)

    ## training dataset result
    yhat_train = model.predict(X_train)
    neyman_pearson_metric_train = neyman_pearson_metric(yhat_train, y_train)
    neyman_pearson_metric_train_array = neyman_pearson_metric(yhat_train, y_train, dict_form = False)
    constraint_violation_train = np.sum(np.maximum( neyman_pearson_metric_train_array - expected_err, 0))
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
    neyman_pearson_metric_test_array = neyman_pearson_metric(yhat_test, y_test, dict_form = False)
    constraint_violation_test = np.sum(np.maximum( neyman_pearson_metric_test_array - expected_err, 0))
    acc_test = accuracy_score(y_test, yhat_test)
    print("neyman pearson result(test):", neyman_pearson_metric_test)
    print("constraint_violation_test:", constraint_violation_test)
    print("overall accuracy(test):", acc_test)
    csvdata["neyman_pearson_metric(test)"] = neyman_pearson_metric_test
    csvdata["constraint_violation(test)"] = constraint_violation_test
    csvdata["accuracy(test)"] = acc_test
    csvdata['expected_err'] = None

    ## model info
    csvdata["model"] = "LGBMClassifier"
    csvdata["time_stamp"] = time_stamp
    csvdata["version"] = version
    csvdata['random_seed'] = seed

    ## save to csv file
    csvfile = open(csvname, 'a', newline='')
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames_np_lgb, delimiter = ",")
    save_to_csv = {**csvdata, **general_params}
    writer.writerow(save_to_csv)
    csvfile.flush()
    csvfile.close()