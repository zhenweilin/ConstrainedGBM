'''LGBMNPClassifier for drybean NeymanPearson task (7-classes)
/* author: Authorname
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import neyman_pearson_metric

# ## read data
datapath = "../../data/DryBeanDataset/"
data = pd.read_excel(datapath + "Dry_Bean_Dataset.xlsx")
X = data.iloc[:, 0:-2]
y = data['Class']
X = X.to_numpy(dtype=np.float32)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y.values)
X = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("**************** LGBMNPClassifier Neyman Pearson Task `drybean NeymanPearson task (7-classes)` ****************")
eps = 1e-8
expected_err = np.array([1-eps, 0.05, 0.05, 0.04, 0.05, 1-eps, 1-eps])
expected_err = expected_err.ravel()
model = lgb.LGBMNPClassifier(expected_err = expected_err)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
res = neyman_pearson_metric(yhat, y_test)
print("neyman pearson result:", res)
print("expected error:", expected_err)
res_array = neyman_pearson_metric(yhat, y_test, dict_form = False)
diff_sum = np.sum(np.maximum(res_array - expected_err, 0))
print("Compare result and expected array (sum):", diff_sum)
print("overall accuracy:", accuracy_score(y_test, yhat))

print("**************** LGBMClassifier Classification (baseline) Task `drybean NeymanPearson task (7-classes)` ****************")
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
res = neyman_pearson_metric(yhat, y_test)
print("neyman pearson result:", res)
print("expected error:", expected_err)
res_array = neyman_pearson_metric(yhat, y_test, dict_form = False)
diff_sum = np.sum(np.maximum(res_array - expected_err, 0))
print("Compare result and expected array (sum):", diff_sum)
print("overall accuracy:", accuracy_score(y_test, yhat))