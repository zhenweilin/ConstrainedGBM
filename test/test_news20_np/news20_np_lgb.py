'''LGBMNPClassifier for news20 NeymanPearson task (20-classes)
/* author: Authorname
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import neyman_pearson_metric
# ## read data
datapath = "../../data/news20/"
X, y = load_svmlight_file(datapath + "news20.scale")
y = y - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("**************** LGBMNPClassifier Neyman Pearson Task `news20 NeymanPearson task (20-classes)` ****************")
expected_err = np.ones([20,1]) - 1e-8
expected_err[16] = 0.02
expected_err[17] = 0.15
expected_err[18] = 0.02
expected_err[19] = 0.02
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

print("**************** LGBMClassifier Classification (baseline) Task `news20 NeymanPearson task (20-classes)` ****************")
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