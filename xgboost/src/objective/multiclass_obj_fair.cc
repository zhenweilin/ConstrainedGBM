/*!
 * Copyright 2018 XGBoost contributors
 * author: Authorname
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj_fair);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "multiclass_obj_fair.cu"
#endif  // XGBOOST_USE_CUDA
