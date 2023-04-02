""" utils function of lightgbm for Neyman Pearson and fairness task  
"""

import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder,  LabelEncoder
from .basic import _log_warning
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

def _initialize_parameters_np(
                sample_weight,
                tau,
                sigma,
                lam,
                theta,
                last_pred_score,
                previous_score,
                expected_err,
                attribute_weight1,
                attribute_weight2,
                X,
                n_class):
    """Initialize parameters for Neyman Pearson Classification

    Parameters
    ----------
    sample_weight
        np.ndarray, sample weight of each instance, set to 1.0 forcely for binary NeymanPearson Classification
    tau
        np.ndarray, stepsize of pirmal variable
    sigma
        np.ndarray, stepsize of dual variable
    lam
        np.ndarray, dual variable
    theta
        np.ndarray, extrapolation stepsize
    last_pred
        np.ndarray, last prediction score
    previous_pred
        np.ndarray, previous prediction score, set to None forcely
    expected_err
        np.ndarray, expected error
    attribute_weight1
        np.ndarray, not for Neyman Pearson Classification, set to None forcely
    attribute_weight2
        np.ndarray, not for Neyman Pearson Classification, set to None forcely
    X
        datapoint
    n_class
        number of classes
    """
    previous_score = None
    attribute_weight1 = None
    attribute_weight2 = None
    ## tau
    if tau is not None:
        assert tau.shape == (1, ), "The shape of tau should be (1, )"
    else:
        tau = np.array([0.1]).ravel()
        
    ## sigma
    if sigma is not None:
        assert sigma.shape == (1, ), "The shape of sigma should be (1, )"
    else:
        sigma = np.array([0.1]).ravel()
        
    ## theta
    if theta is not None:
        assert theta.shape == (1, ), "The shape of theta should be (1, )"
    else:
        theta = np.array([1.0]).ravel()
        
    (m, _) = X.shape
    if n_class == 2:
        ## sample_weight
        sample_weight = np.ones([m, 1]).ravel()
        ## lam
        if lam is not None:
            assert lam.shape == (1, )
        else:
            lam = np.array([0.1]).ravel()
            
        ## theta
        if theta is not None:
            assert theta.shape == (1, )
        else:
            theta = np.array([1.0]).ravel()
            
        ## last_pred_score
        if last_pred_score is not None:
            assert last_pred_score.shape == (m, )
        else:
            last_pred_score = np.zeros([m, 1]).ravel() # 1/(exp(-f) + 1) = 0.5 
            
        ## expected_err
        if expected_err is not None:
            assert expected_err.shape == (1, )
        else:
            expected_err = np.array([0.1]).ravel()
            
    else:
        ## class_weight_w
        if sample_weight is not None:
            assert sample_weight.shape == (m, )
        else:
            sample_weight = np.ones([m, 1]).ravel()
        
        ## lam
        if lam is not None:
            assert lam.shape == (n_class, )
        else:
            lam = np.ones([n_class, 1]).ravel()
            
        ## last_pred_score
        if last_pred_score is not None:
            assert last_pred_score.shape == (m, n_class)
        else:
            last_pred_score = np.zeros([m, n_class])
        
        ## expected_err
        if expected_err is not None:
            assert expected_err.shape == (n_class, ), f"Desired expected_err shape is ({n_class}, )"
        else:
            expected_err = (np.ones([n_class, 1]) * 0.1).ravel()
    return (sample_weight,
            tau,
            sigma,
            lam,
            theta,
            last_pred_score,
            previous_score,
            expected_err,
            attribute_weight1,
            attribute_weight2)
    
def _construct_sensitive_attribute(X, sensitive_attribute):
    (m, nFeature) = X.shape
    if sensitive_attribute is not None:
        if isinstance(sensitive_attribute, int) or  sensitive_attribute.shape == (1, ):
            sensitive_attribute = int(sensitive_attribute)
            assert sensitive_attribute < nFeature, f"Sensitive_attribute feature index is greater than feature number of dataset, the number of feature is {nFeature}."
            sensitive_attribute = X[:, sensitive_attribute]
        else:
            assert sensitive_attribute.ravel().shape == (m, ), "The dimension of `sensitive_attribute` is not match that of X"
    else:
        raise ValueError("Sensitive_attribute is None, which needs to be set!")
    encoderLabel = LabelEncoder().fit(sensitive_attribute.ravel())
    sensitive_attribute = encoderLabel.transform(sensitive_attribute).reshape(-1, 1).ravel()
    return sensitive_attribute
    
    
def _initialize_parameters_fair(
                sample_weight,
                tau,
                sigma,
                lamArray,
                theta,
                last_pred_score,
                previous_score,
                attribute_weight1,
                attribute_weight2,
                alpha_max,
                sensitive_attribute,
                X,
                n_class):
    """Initialize parameters for Fairness Classification

    Parameters
    ----------
    sample_weight
        np.ndarray, sample weight of each instance, set to 1.0 forcely for binary NeymanPearson Classification
    tau
        np.ndarray, stepsize of pirmal variable
    sigma
        np.ndarray, stepsize of dual variable
    lamArray
        np.ndarray, dual variable
    theta
        np.ndarray, extrapolation stepsize
    last_pred_score
        np.ndarray, last prediction score
    previous_score
        np.ndarray, previous prediction score, set to None forcely
    attribute_weight1
        np.ndarray, see `doc` for specific meaning
    attribute_weight2
        np.ndarray, see `doc` for specific meaning
    sensitive_attribute
        np.ndarray or int, specify the sensitive_attribute feature
    X
        dataset
    n_class
        number of classes
    """
    ## fair gbm initialize
    (m, nFeature) = X.shape
    ## sensitive_attribute
    if sensitive_attribute is not None:
        if isinstance(sensitive_attribute, int) or  sensitive_attribute.shape == (1, ):
            sensitive_attribute = int(sensitive_attribute)
            assert sensitive_attribute < nFeature, f"Sensitive_attribute feature index is greater than feature number of dataset, the number of feature is {nFeature}."
            sensitive_attribute = X[:, sensitive_attribute]
        else:
            assert sensitive_attribute.ravel().shape == (m, ), "The dimension of `sensitive_attribute` is not match that of X"
    else:
        raise ValueError("Sensitive_attribute is None, which needs to be set!")
    
    # sensitive_attribute = sensitive_attribute.reshape(-1,1)
    encoderLabel = LabelEncoder().fit(sensitive_attribute.ravel())
    sensitive_attribute = encoderLabel.transform(sensitive_attribute).reshape(-1, 1)
    
    encoder = OneHotEncoder().fit(sensitive_attribute)
    sensitive_attributeOnehot = encoder.transform(sensitive_attribute).toarray()
    attribute_stat = np.sum(sensitive_attributeOnehot, axis = 0).ravel()
    attribute_max = np.max(attribute_stat)
    attribute_min = np.min(attribute_stat)
    n_attribute = len(np.unique(sensitive_attribute))
    assert n_attribute != 1, "The length of attribute is 1, no fairness task!"
    
    ## tau
    if tau is not None:
        assert tau.shape == (1, )
    else:
        tau = np.array([0.1]).ravel()
        
    ## sigma
    if sigma is not None:
        assert sigma.shape == (1, )
    else:
        sigma = np.array([0.1]).ravel()
        
    ## theta
    if theta is not None:
        assert theta.shape == (1, )
    else:
        theta = np.array([1.0]).ravel()
        
    ## lam
    if lamArray is not None:
        assert lamArray.shape == (n_attribute, n_attribute), f"The right dimension of lam is (n_attribute, n_attribute)"
    else:
        lamArray = np.ones([n_attribute, n_attribute]) - np.eye(n_attribute, n_attribute)
    ## attribute_weight1
    if attribute_weight1 is not None:
        assert attribute_weight1.shape == (m, )
    else:
        attribute_weight1 = np.ones([m, 1]).ravel()
        
    ## attribute_weight2
    if attribute_weight2 is not None:
        assert attribute_weight2.shape == (m, )
    else:
        attribute_weight2 = np.array([(1/4 + attribute_max / 4 * (n_attribute * (n_attribute - 1)))])
    
    ## alphaArray
    alphaArray = 0.00001 * (np.ones([n_attribute, n_attribute]) - np.eye(n_attribute))

    ## alphaArrayMax
    alphaArrayMax = alpha_max * (np.ones([n_attribute, n_attribute]) - np.eye(n_attribute))
    
    if n_class == 2:
        ## sample_weight
        if sample_weight is not None:
            assert sample_weight.shape == (m, )
        else:
            sample_weight = np.ones([m, 1]).ravel()
            
        ## last_pred_score
        if last_pred_score is not None:
            assert last_pred_score.shape == (m, )
        else:
            last_pred_score = np.zeros([m, 1]).ravel() # 1/(exp(-f) + 1) = 0.5
        
        ## previous_score
        if previous_score is not None:
            assert previous_score.shape == (m, )
        else:
            previous_score = np.zeros([m, 1]).ravel()
    else:
        ## sample_weight
        if sample_weight is not None:
            assert sample_weight.shape == (m, )
        else:
            sample_weight = np.ones([m, 1]).ravel()
            
        ## last_pred_score
        if last_pred_score is not None:
            assert last_pred_score.shape == (m, n_class)
        else:
            last_pred_score = np.zeros([m, n_class]).ravel()
        
        ## previous_score
        if previous_score is not None:
            assert previous_score.shape == (m, n_class)
        else:
            previous_score = np.zeros([m, n_class]).ravel()
    return (sample_weight,
            tau,
            sigma,
            lamArray,
            theta,
            last_pred_score,
            previous_score,
            alphaArray,
            alphaArrayMax,
            attribute_weight1,
            attribute_weight2,
            sensitive_attribute,
            attribute_stat,
            attribute_max,
            attribute_min)
    
def _construct_alpha_np(expected_err,
                     y,
                     n_class,
                     initial = True,
                     prob = None):
    """Construct alpha value in Neyman Pearson Classification

    Parameters
    ----------
    expected_err
        np.ndarray, every expected error rate (expected error rate for 0 class when biary classification)
    y
        np.ndarray, label value
    n_class
        number of classes
    initial
        bool
    prob
        np.ndarray, predicted probability
    """
    if initial:
        if n_class == 2:
            ## first construct alpha, default use alpha = n0 * expected_err * np.log(n_class)
            n0 = len(y) - sum(y)
            alpha = n0 * expected_err * np.log(n_class)
        else:
            nk_arr = np.bincount(y)
            alpha = nk_arr * expected_err * np.log(n_class)
            
    else:
        if n_class == 2:
            # ## alpha = -expected_err * sum(( 1 - y )*log( 1 - haty ))
            alpha = -expected_err * np.sum((1-y) * np.log(prob[:,0]))
        else:
            labelarray = y.reshape(-1,1)
            encoder = OneHotEncoder().fit(labelarray)
            labelOnehot = encoder.transform(labelarray).toarray()
            loss = np.sum(np.log(prob) * labelOnehot, axis = 0)
            alpha = -(expected_err * loss)
    return alpha


def _calculate_constraint_loss(n_class, prob, label, labelOnehot=None):
    """Calculate the constraint loss for extrapolation

    Parameters
    ----------
    n_class
        number of classes
    prob
        np.ndarray, predicted probability
    label
        np.ndarray, label value
    labelOnehot
        np.ndarray, one hot encoding of label for multi-class
    """
    if n_class == 2:
        prob = prob[:, 1]
        return sum(-1 * np.log(1 - prob) * (1 - label))
    else:
        return sum(-np.log(prob) * labelOnehot)
    

def delete_outlier_data(X,
                        label,
                        last_raw_predt,
                        prob,
                        topOneTenthIdx,
                        topOneTenthLoss,
                        outlier_threshold,
                        lossMean9):
    """Delete outlier data to make data better

    Parameters
    ----------
    X
        dataset
    label
        np.ndarray, label value
    last_raw_predt
        np.ndarray, predict raw value (without softmax or sigmoid)
    prob
        np.ndarray, predicted probability
    topOneTenthIdx
        np.ndarray, Top 10% of indexes sorted by loss
    topOneTenthLoss
        np.ndarray, Top 10% of loss sorted by it
    outlier_threshold
        np.ndarray, threshold to determine outlier
    lossMean9
        np.ndarray, 90% of the mean value after sorting by loss size
    """
    outlierIdx = topOneTenthIdx[topOneTenthLoss > outlier_threshold * lossMean9]
    if len(outlierIdx) > 0:
        warnings.warn(f"The outlier dataset size is:{len(outlierIdx)}.\nThe outlierIdx is:{outlierIdx}")
    if not isinstance(X, (csr_matrix, csr_matrix, coo_matrix)):
        X = np.delete(X, outlierIdx, axis=0)
    else:
        rows_to_keep = np.array([i for i in range(X.shape[0]) if i not in outlierIdx])
        X = X[rows_to_keep, :]
    label = np.delete(label, outlierIdx, axis=0)
    last_raw_predt = np.delete(last_raw_predt, outlierIdx, axis=0)
    prob = np.delete(prob, outlierIdx, axis=0)
    return (X, label, last_raw_predt, prob)
    
    
def _outlier_detection(X,
                       prob,
                       label,
                       last_raw_predt,
                       n_class,
                       outlier_threshold,
                       labelOnehot):
    """Delete outlier data to make data better

    Parameters
    ----------
    X
        dataset
    prob
        np.ndarray, predict probability
    label
        np.ndarray
    last_raw_predt
        np.ndarray, predict raw value (without softmax or sigmoid)
    n_class
        number of classes
    outlier_threshold
        np.ndarray, threshold to determine outlier
    labelOnehot
        np.ndarray, one hot encoding of label for multi-class
    """
    if n_class == 2:
        lossData = -label * np.log(prob[:, 1]) - (1 - label) * np.log(prob[:,0])
    else:
        lossData = -np.sum(labelOnehot * np.log(prob), axis = 1)
    topOneTenth = np.ceil(X.shape[0] / 10).astype(int)
    rankLossIdx = lossData.argsort()[::-1]
    topOneTenthIdx = rankLossIdx[0:topOneTenth]
    tailNineTenthsIdx = rankLossIdx[topOneTenth:]
    lossMean9 = np.mean(lossData[tailNineTenthsIdx])
    topOneTenthLoss = lossData[topOneTenthIdx]
    
    (X_copy, label_copy, last_raw_predt_copy, prob_copy) = (X, label, last_raw_predt, prob)
    
    (X, label, last_raw_predt, prob) = delete_outlier_data(X_copy, label_copy, last_raw_predt_copy, prob_copy, topOneTenthIdx, topOneTenthLoss, outlier_threshold, lossMean9)

    n_class_new = len(np.unique(label))
    count = 1
    while n_class_new != n_class:
        outlier_threshold = outlier_threshold * 1.5
        (X, label, last_raw_predt, prob) = delete_outlier_data(X_copy, label_copy, last_raw_predt_copy, prob_copy, topOneTenthIdx, topOneTenthLoss, outlier_threshold, lossMean9)
        count = count + 1
        if count == 5: raise ValueError(f"Increase `outlier_threshold` to decrease outlier data point, since class number is different! Now the outlier_threshold:{outlier_threshold}")
    return (X, last_raw_predt, label, prob)

def _construct_attribute(lamArray, attribute_stat, sensitive_attribute, scale_factor):
    """Construct `attribute_weight1` and `attribute_weight2`

    Parameters
    ----------
    lamArray
        np.ndarray, lambda array for fairness
    attribute_stat
        np.ndarray, sensitive attribute data point number statistics
    sensitive_attribute
        np.ndarray
    scale_factor
        np.ndarray, attribute number min in doc
    """
    row_sum = np.sum(lamArray, axis=1).ravel()
    col_sum = np.sum(lamArray, axis=0).ravel()
    
    attribute_weight1_vec = scale_factor * (row_sum - col_sum) / attribute_stat + 1
    attribute_weight1 = attribute_weight1_vec[sensitive_attribute]
    
    all_sum = np.sum(lamArray)
    rho = 0.25
    attribute_weight2 = np.array([(rho/2 + rho * all_sum * scale_factor / 2 )])
    return (attribute_weight1.ravel(), attribute_weight2.ravel())

def _calculate_constraint_loss_fair(n_class,
                                    prob,
                                    label,
                                    sensitive_attribute_Onehot,
                                    attribute_stat,
                                    labelOnehot=None):
    """Calculate constraint loss of fairness objective

    Parameters
    ----------
    n_class
        number of classes
    prob
        np.ndarray, predict probability
    label
        np.ndarray
    sensitive_attribute_Onehot
        np.ndarray, one hot encoding of sensitive attribute
    attribute_stat
        np.ndarray, sensitive attribute data point number statistics
    labelOnehot
        np.ndarray, one hot encoding of label for multi-class
    """
    '''
    \sum_i \ell(f,y|S=a) / n_a 
    '''
    if n_class == 2:
        prob = prob[:, 1]
        lossdata = -1 * (label * np.log(prob) + np.log(1 - prob) * (1 - label))
    else:
        lossdata = np.sum(-np.log(prob) * labelOnehot, axis = 1)

    lossdata = lossdata.reshape(-1, 1)
    lossdata = np.tile(lossdata, sensitive_attribute_Onehot.shape[1])
    loss_attribute = np.sum(lossdata * sensitive_attribute_Onehot, axis = 0).ravel()
    loss_attribute = loss_attribute / attribute_stat
    return loss_attribute
        
def _construct_zArray(loss_attribute_old,
                      loss_attribute_new,
                      proximal_penalty_old,
                      proximal_penalty_new,
                      alphaArray, theta,
                      scale_factor):
    """Calculate z (extrapolation function value)

    Parameters
    ----------
    loss_attribute_old
        np.ndarray, old loss value classified by attribute
    loss_attribute_new
        np.ndarray, new loss value classified by attribute
    scale_factor
        np.ndarray, factor for constraint and objective balance
    proximal_penalty_old
        np.ndarray, old penalty term
    proximal_penalty_new
        np.ndarray, new penalty term
    alphaArray
        np.ndarray, alpha array
    theta
        np.ndarray, extrapolation stepsize
    """
    loss_attribute_new_col = loss_attribute_new.reshape(-1, 1)
    loss_attribute_new_row = loss_attribute_new.reshape(1, -1)
    diffArray1 = scale_factor * (loss_attribute_new_col - loss_attribute_new_row + proximal_penalty_new)
    
    loss_attribute_old_col = loss_attribute_new.reshape(-1, 1)
    loss_attribute_old_row = loss_attribute_old.reshape(1, -1)
    diffArray2 = scale_factor * (loss_attribute_old_col - loss_attribute_old_row + proximal_penalty_old)
    zArray = diffArray1 - scale_factor * alphaArray + theta * (diffArray1 - diffArray2)
    row, col = np.diag_indices_from(zArray)
    zArray[row, col] = 0
    return zArray

def _calculate_proximal_term(last_pred_score, previous_score):
    """Calculate penalty term
        rho * norm(f-f^l, 2)/2
    """  
    return np.power(np.linalg.norm(last_pred_score - previous_score, 2), 2) / (8)


def fairness_metric(yhat, y_test, sensitive_attribute, dict_form = True):
    ''' Compute the fairness result
    '''
    yhat = yhat.ravel()
    if not isinstance(y_test, np.ndarray):
        ## assume pd.Dataframe
        y_test = y_test.values.ravel()
    encoder = LabelEncoder()
    sensitive_attribute = encoder.fit(sensitive_attribute.ravel()).transform(sensitive_attribute.ravel())
    sensitive_attribute = sensitive_attribute.ravel()
    assert (yhat.shape == y_test.shape and y_test.shape == sensitive_attribute.shape), "The input three args shape should be the same"
    n_sensitive = len(np.unique(sensitive_attribute))
    if dict_form:
        res = {}
        for i in range(n_sensitive):
            if len(y_test[sensitive_attribute == i]) == 0:
                _log_warning(f"The length of y_test with slice sensitive_attribute {i} is 0")
                res[encoder.inverse_transform([i]).item(0)] = -1
            else:
                res[encoder.inverse_transform([i]).item(0)] = np.sum(yhat[sensitive_attribute == i] != y_test[sensitive_attribute == i])/len(y_test[sensitive_attribute == i])
    else:
        res = np.zeros([n_sensitive])
        for i in range(n_sensitive):
            if len(y_test[sensitive_attribute == i]) == 0:
                _log_warning(f"The length of y_test with slice sensitive_attribute {i} is 0")
                res[i] = -1
            else:
                res[i] = np.sum(yhat[sensitive_attribute == i] != y_test[sensitive_attribute == i])/len(y_test[sensitive_attribute == i])
    return res


def neyman_pearson_metric(yhat, y_test, dict_form = True):
    ''' Compute the neyman pearson result
    '''
    yhat = yhat.ravel()
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values.ravel()
    assert (yhat.shape == y_test.shape), "The input two args shape should be the same"
    n_class = len(np.unique(y_test))
    assert (n_class > 1), "The number of classes in test data should greater than 1, at least 2"
    if dict_form:
        res = {}
    else:
        res = np.zeros([n_class])
        
    for i in range(n_class):
        res[i] = np.sum(yhat[y_test == i] != y_test[y_test == i])/len(y_test[y_test == i])
    return res if dict_form else res.ravel()

def _construct_Lambda_matrix(Lambda_array, lamb_vec, i):
    """Construct Lambda matrix

    Parameters
    ----------
    Lambda_array
        np.ndarray, Lambda array
    lamb_vec
        np.ndarray, lambda vector
    i
        int, index of iteration number
    """
    if i < 5:
        ## put into Lambda_array directly
        Lambda_array[:, i] = lamb_vec.ravel()
    else:
        ## put into Lambda_array, and delte the first column
        Lambda_array[:, :-1] = Lambda_array[:, 1:]
        Lambda_array[:, -1] = lamb_vec
    return Lambda_array

def _heuristic_terminate_apd(Lambda_array, i):
    """Heuristic termination condition for APD

    Parameters
    ----------
    Lambda_array
        np.ndarray, Lambda array
    i
        int, index of iteration number
    """
    if i < 5:
        return False
    else:
        lamb_max = np.max(Lambda_array, axis = 1)
        lamb_min = np.min(Lambda_array, axis = 1)
        lamb_mean = np.mean(Lambda_array, axis = 1)

        if np.linalg.norm((lamb_max - lamb_min) / 2 - lamb_mean) / (1 + np.linalg.norm(lamb_mean)) < 1e-3:
            return True
        else:
            return False
