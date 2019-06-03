// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <string>
#include <utility>
#include <vector>

#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3);

  if (!this->layer_param_.loss_param().has_normalization() &&
    this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
    LossParameter_NormalizationMode_VALID :
    LossParameter_NormalizationMode_BATCH_SIZE;
  }
  else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }

  outer_num_ = bottom[0]->num();
  inner_num_ = bottom[0]->height() * bottom[0]->width();

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  // top[2] stores per-instance loss, which takes the shape of N*1*H*W
  if (top.size()>=2) {
    top[1]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
Dtype SmoothL1LossLayer<Dtype>::get_normalizer(
  LossParameter_NormalizationMode normalization_mode, Dtype pre_fixed_normalizer) {
  Dtype normalizer;
  switch (normalization_mode) {
  case LossParameter_NormalizationMode_FULL:
    normalizer = Dtype(outer_num_ * inner_num_);
    break;
  case LossParameter_NormalizationMode_VALID:
    normalizer = Dtype(outer_num_ * inner_num_);
    break;
  case LossParameter_NormalizationMode_BATCH_SIZE:
    normalizer = Dtype(outer_num_);
    break;
  case LossParameter_NormalizationMode_PRE_FIXED:
    normalizer = pre_fixed_normalizer;
    break;
  case LossParameter_NormalizationMode_NONE:
    normalizer = Dtype(1);
    break;
  default:
    LOG(FATAL) << "Unknown normalization mode: "
      << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
