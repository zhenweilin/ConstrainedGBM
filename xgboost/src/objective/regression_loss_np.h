/*!
 * Copyright 2017-2022 XGBoost contributors
 * author: Authorname
 */

#include <dmlc/omp.h>
#include <xgboost/logging.h>

#include <cmath>

#include "../common/math.h"
#include "xgboost/task.h"  // ObjInfo

namespace xgboost {
namespace obj {
// common regressions
// linear regression

// logistic loss for probability regression task
struct LogisticNPRegression {
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return common::Sigmoid(x); }
  XGBOOST_DEVICE static bool CheckLabel(bst_float x) { return x >= 0.0f && x <= 1.0f; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label, bst_float class_weight, bst_float tau, bst_float last_pred) {
    bst_float probt;
    probt = common::Sigmoid(predt);
    // class_weight is lambda
    return probt  * (label + class_weight * (1-label)) - label + (predt - last_pred) * 1.0f/tau;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label, bst_float class_weight, bst_float tau, bst_float last_pred) {
    const float eps = 1e-16f;
    predt = common::Sigmoid(predt);
    return fmaxf((label + class_weight * (1-label)) * predt * (1.0f - predt) + 1.0f/tau, eps);
  }
  static bst_float ProbToMargin(bst_float base_score) {
    CHECK(base_score > 0.0f && base_score < 1.0f)
        << "base_score must be in (0,1) for logistic loss, got: " << base_score;
    return -logf(1.0f / base_score - 1.0f);
  }
  static const char* LabelErrorMsg() { return "label must be in [0,1] for logistic regression"; }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:logistic_np"; }

  static ObjInfo Info() { return ObjInfo::kRegression; }
};

// logistic loss for binary classification task
struct LogisticNPClassification : public LogisticNPRegression {
  static const char* DefaultEvalMetric() { return "logloss"; }
  static const char* Name() { return "binary:logistic_np"; }
  static ObjInfo Info() { return ObjInfo::kBinary; }
};

// logistic loss, but predict un-transformed margin
// struct LogisticNPRaw : public LogisticNPRegression {
//   XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
//   XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label, bst_float class_weight, bst_float tau, bst_float last_pred) {
//     bst_float probt;
//     probt = common::Sigmoid(predt);
//     return probt  * (label + class_weight * (1-label)) - label + (predt - last_pred) * 1.0f/tau;
//   }
//   XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label, bst_float class_weight, bst_float tau, bst_float last_pred) {
//     const float eps = 1e-16f;
//     predt = common::Sigmoid(predt);
//     return fmaxf((label + class_weight * (1-label)) * predt * (1.0f - predt) + 1.0f/tau, eps);
//   }
//   static bst_float ProbToMargin(bst_float base_score) { return base_score; }
//   static const char* DefaultEvalMetric() { return "logloss_np"; }

//   static const char* Name() { return "binary:logitraw_np"; }

//   static ObjInfo Info() { return ObjInfo::kRegression; }
// };

}  // namespace obj
}  // namespace xgboost
