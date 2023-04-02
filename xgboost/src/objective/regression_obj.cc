/*!
 * Copyright 2018 XGBoost contributors
 * Modified by Authorname
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj);
DMLC_REGISTRY_FILE_TAG(regression_obj_fair);
DMLC_REGISTRY_FILE_TAG(regression_obj_np);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "regression_obj.cu"
#endif  // XGBOOST_USE_CUDA
