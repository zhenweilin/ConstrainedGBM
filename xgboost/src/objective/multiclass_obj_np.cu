/*!
 * Copyright 2015-2022 by XGBoost Contributors
 * \file multi_class_np.cu
 * \brief Definition of multi-class Neyman Pearson classification objectives.
 * \author Authorname
 */
#include <dmlc/omp.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

#include "xgboost/parameter.h"
#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/json.h"

#include "../common/common.h"
#include "../common/math.h"
#include "../common/transform.h"

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(multiclass_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct SoftmaxMultiClassNPParam : public XGBoostParameter<SoftmaxMultiClassNPParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassNPParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

class SoftmaxMultiClassNPObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassNPObj(bool output_prob)
  : output_prob_(output_prob) {}

  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
  }

  ObjInfo Task() const override { return ObjInfo::kClassification; }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    // Remove unused parameter compiler warning.
    (void) iter;
    if (info.labels.Size() == 0) {
      return;
    }
    CHECK(preds.Size() == (static_cast<size_t>(param_.num_class) * info.labels.Size()))
        << "SoftmaxMultiClassNPObj: label size and pred size does not match.\n"
        << "label.Size() * num_class: "
        << info.labels.Size() * static_cast<size_t>(param_.num_class) << "\n"
        << "num_class: " << param_.num_class << "\n"
        << "preds.Size(): " << preds.Size();
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(preds.Size() / nclass);

    auto device = ctx_->gpu_id;
    out_gpair->SetDevice(device);
    info.labels.SetDevice(device);
    // info.weights_.SetDevice(device);
    info.class_weight.SetDevice(device);
    info.last_pred.SetDevice(device);
    info.tau.SetDevice(device);
    preds.SetDevice(device);

    label_correct_.Resize(1);
    label_correct_.SetDevice(device);
    out_gpair->Resize(preds.Size());
    label_correct_.Fill(1);
    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t idx,
                           common::Span<GradientPair> gpair,
                           common::Span<bst_float const> labels,
                           common::Span<bst_float const> preds,
                           common::Span<int> _label_correct,
                           common::Span<bst_float const> class_weight, 
                           common::Span<bst_float const> tau,
                           common::Span<bst_float const> last_pred) {
          common::Span<bst_float const> point = preds.subspan(idx * nclass, nclass);
          // Part of Softmax function
          bst_float wmax = std::numeric_limits<bst_float>::min();
          for (auto const i : point) { wmax = fmaxf(i, wmax); }
          double wsum = 0.0f;
          for (auto const i : point) { wsum += expf(i - wmax); }
          auto label = labels[idx];
          if (label < 0 || label >= nclass) {
            _label_correct[0] = 0;
            label = 0;
          }

          for (int k = 0; k < nclass; ++k) {
            // Computation duplicated to avoid creating a cache.
            // class_weight is `lambda + w_k`
            bst_float p = expf(point[k] - wmax) / static_cast<float>(wsum);
            const bst_float h = p * (1.0f - p) * class_weight[label] + 1.0f/tau[0];
            p = p * class_weight[label] + 1.0f/tau[0] * (point[k] - last_pred[idx * nclass + k]);
            p = label == k ? p - class_weight[k] : p;
            gpair[idx * nclass + k] = GradientPair(p, 2 * h);
          }
        }, common::Range{0, ndata}, ctx_->Threads(), device)
        .Eval(out_gpair, info.labels.Data(), &preds, &label_correct_, &info.class_weight, &info.tau, &info.last_pred);

    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag != 1) {
        LOG(FATAL) << "SoftmaxMultiClassNPObj: label must be in [0, num_class).";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float>* io_preds) const override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "mlogloss";
  }

  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) const {
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(io_preds->Size() / nclass);

    auto device = io_preds->DeviceIdx();
    if (prob) {
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
            common::Span<bst_float> point =
                _preds.subspan(_idx * nclass, nclass);
            common::Softmax(point.begin(), point.end());
          },
          common::Range{0, ndata}, this->ctx_->Threads(), device)
          .Eval(io_preds);
    } else {
      io_preds->SetDevice(device);
      HostDeviceVector<bst_float> max_preds;
      max_preds.SetDevice(device);
      max_preds.Resize(ndata);
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<const bst_float> _preds,
                             common::Span<bst_float> _max_preds) {
            common::Span<const bst_float> point =
                _preds.subspan(_idx * nclass, nclass);
            _max_preds[_idx] =
                common::FindMaxIndex(point.cbegin(), point.cend()) -
                point.cbegin();
          },
          common::Range{0, ndata}, this->ctx_->Threads(), device)
          .Eval(io_preds, &max_preds);
      io_preds->Resize(max_preds.Size());
      io_preds->Copy(max_preds);
    }
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    if (this->output_prob_) {
      out["name"] = String("multi:softprob_np");
    } else {
      out["name"] = String("multi:softmax_np");
    }
    out["softmax_np_multiclass_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["softmax_np_multiclass_param"], &param_);
  }

 private:
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassNPParam param_;
  // Cache for max_preds
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassNPParam);

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClassNP, "multi:softmax_np")
.describe("Softmax for multi-class classification, output class index.")
.set_body([]() { return new SoftmaxMultiClassNPObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobNPMultiClassNP, "multi:softprob_np")
.describe("Softmax for multi-class classification, output probability distribution.")
.set_body([]() { return new SoftmaxMultiClassNPObj(true); });
}  // namespace obj
}  // namespace xgboost
