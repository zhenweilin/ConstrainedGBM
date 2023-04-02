import csv

# ## write csv header to file `fair_lgb.csv`
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


csvfile_path = "./fair_lgb.csv"
with open(csvfile_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames = fieldnames_fair_lgb, delimiter=',')
    writer.writeheader()


# ## write csv header to file `fair_xgb.csv`
fieldnames_fair_xgb = [
    "dataset",\
    "binary/multi",\
    "model",\
    "tau",\
    'sigma',\
    "outlier_threshold",\
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
csvfile_path = "./fair_xgb.csv"
with open(csvfile_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames = fieldnames_fair_xgb, delimiter=',')
    writer.writeheader()


# ## write csv header to file `np_lgb.csv`
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

csvfile_path = "./np_lgb.csv"
with open(csvfile_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames = fieldnames_np_lgb, delimiter=',')
    writer.writeheader()



# ## write csv header to file `np_xgb.csv`
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

csvfile_path = "./np_xgb.csv"
with open(csvfile_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames = fieldnames_np_xgb, delimiter=',')
    writer.writeheader()
