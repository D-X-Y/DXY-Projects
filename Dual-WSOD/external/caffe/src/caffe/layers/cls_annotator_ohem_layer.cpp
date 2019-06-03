// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/rfcn_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  // use for ohem with only scores
  template <typename Dtype>
  void ClsAnnotatorOHEMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    BoxAnnotatorOHEMParameter box_anno_param = this->layer_param_.box_annotator_ohem_param();
    roi_per_img_ = box_anno_param.roi_per_img();
    CHECK_GT(roi_per_img_, 0);
    ignore_label_ = box_anno_param.ignore_label();
  }

  template <typename Dtype>
  void ClsAnnotatorOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();                   // rois
    CHECK_EQ(5, bottom[0]->channels());
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    spatial_dim_ = height_*width_;
    
    CHECK_EQ(bottom[1]->num(), num_);          // per_roi_loss
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), height_);
    CHECK_EQ(bottom[1]->width(), width_);

    CHECK_EQ(bottom[2]->num(), num_);          // labels
    CHECK_EQ(bottom[2]->channels(), 1);
    CHECK_EQ(bottom[2]->height(), height_);
    CHECK_EQ(bottom[2]->width(), width_);

    // Labels for scoring
    top[0]->Reshape(num_, 1, height_, width_);
  }

  template <typename Dtype>
  void ClsAnnotatorOHEMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void ClsAnnotatorOHEMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


#ifdef CPU_ONLY
  STUB_GPU(ClsAnnotatorOHEMLayer);
#endif

  INSTANTIATE_CLASS(ClsAnnotatorOHEMLayer);
  REGISTER_LAYER_CLASS(ClsAnnotatorOHEM);

}  // namespace caffe
