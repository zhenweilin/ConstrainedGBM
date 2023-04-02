import numpy as np
def generate_random_paras_fair_lgb(seed):
    np.random.seed(seed)
    general_params = {
    "n_estimators": np.random.randint(50, 200, [1,1]).ravel()[0],
    "learning_rate": np.random.uniform(0.001, 0.9, [1,1]).ravel()[0],
    "num_leaves": np.random.randint(30, 100, [1,1]).ravel()[0],
    "min_child_weight": np.random.uniform(0.01, 0.9, [1,1]).ravel()[0],
    "max_depth": np.random.randint(4, 30, [1,1]).ravel()[0],
    "colsample_bytree": np.random.uniform(0.001, 1, [1,1]).ravel()[0],
    "reg_alpha": np.random.uniform(0, 1, [1,1]).ravel()[0],
    "reg_lambda": np.random.uniform(0, 1, [1,1]).ravel()[0],
}
    scale_factor_seed = np.random.uniform(0,1,[1,1]).ravel()[0]
    params = {
    "tau" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "sigma" : np.random.uniform(0.1, 0.3, [1,1]).ravel(),
    "outlier_threshold" : np.random.randint(7, 30, [1,1]).ravel(),
    "max_iter_T": np.random.randint(30, 50, [1,1]).ravel()[0],
    "max_iter_L": np.random.randint(20, 40, [1,1]).ravel()[0],
    "scale_factor": np.random.uniform(0.5, 10, [1,1]).ravel() if scale_factor_seed < 0.7 else None,
    "alpha_max": np.random.uniform(0.001, 0.2, [1,1]).ravel(),
}
    return (general_params, params)


def generate_random_paras_fair_xgb(seed):
    np.random.seed(seed)
    general_params = {
    "n_estimators": np.random.randint(50, 200, [1,1]).ravel()[0],
    "learning_rate": np.random.uniform(0.1, 0.3, [1,1]).ravel()[0],
    "max_leaves": np.random.randint(30, 100, [1,1]).ravel()[0],
    "min_child_weight": np.random.uniform(0.01, 0.9, [1,1]).ravel()[0],
    "max_depth": np.random.randint(4, 30, [1,1]).ravel()[0],
    "colsample_bytree": np.random.uniform(0.001, 1, [1,1]).ravel()[0],
    "reg_alpha": np.random.uniform(0, 1, [1,1]).ravel()[0],
    "reg_lambda": np.random.uniform(0, 1, [1,1]).ravel()[0],
    "gamma": np.random.uniform(0, 2, [1,1]).ravel()[0],
}
    scale_factor_seed = np.random.uniform(0,1,[1,1]).ravel()[0]
    params = {
    "tau" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "sigma" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "outlier_threshold" : np.random.randint(7, 30, [1,1]).ravel(),
    "max_iter_T": np.random.randint(30, 50, [1,1]).ravel()[0],
    "max_iter_L": np.random.randint(20, 40, [1,1]).ravel()[0],
    "scale_factor": np.random.uniform(0.5, 10, [1,1]).ravel() if scale_factor_seed < 0.7 else None,
    "alpha_max": np.random.uniform(0.001, 0.2, [1,1]).ravel(),
}
    return (general_params, params)

def generate_random_paras_np_lgb(seed):
    np.random.seed(seed)
    general_params = {
    "n_estimators": np.random.randint(50, 200, [1,1]).ravel()[0],
    "learning_rate": np.random.uniform(0.001, 0.9, [1,1]).ravel()[0],
    "num_leaves": np.random.randint(30, 100, [1,1]).ravel()[0],
    "min_child_weight": np.random.uniform(0.01, 0.9, [1,1]).ravel()[0],
    "max_depth": np.random.randint(4, 30, [1,1]).ravel()[0],
    "colsample_bytree": np.random.uniform(0.001, 1, [1,1]).ravel()[0],
    "reg_alpha": np.random.uniform(0, 1, [1,1]).ravel()[0],
    "reg_lambda": np.random.uniform(0, 1, [1,1]).ravel()[0],
}
    params = {
    "tau" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "sigma" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "outlier_threshold" : np.random.randint(7, 30, [1,1]).ravel(),
    "max_iter_T": np.random.randint(30, 100, [1,1]).ravel()[0],
}
    return (general_params, params)

def generate_random_paras_np_xgb(seed):
    np.random.seed(seed)
    general_params = {
    "n_estimators": np.random.randint(50, 200, [1,1]).ravel()[0],
    "learning_rate": np.random.uniform(0.001, 0.9, [1,1]).ravel()[0],
    "max_leaves": np.random.randint(30, 100, [1,1]).ravel()[0],
    "min_child_weight": np.random.uniform(0.01, 0.9, [1,1]).ravel()[0],
    "max_depth": np.random.randint(4, 30, [1,1]).ravel()[0],
    "colsample_bytree": np.random.uniform(0.001, 1, [1,1]).ravel()[0],
    "reg_alpha": np.random.uniform(0, 1, [1,1]).ravel()[0],
    "reg_lambda": np.random.uniform(0, 1, [1,1]).ravel()[0],
    "gamma": np.random.uniform(0, 2, [1,1]).ravel()[0],
}
    params = {
    "tau" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "sigma" : np.random.uniform(0.001, 0.2, [1,1]).ravel(),
    "outlier_threshold" : np.random.randint(7, 30, [1,1]).ravel(),
    "max_iter_T": np.random.randint(30, 100, [1,1]).ravel()[0],
}
    return (general_params, params)