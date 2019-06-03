#include "caffe/layers/pad_channel_layer.hpp"

namespace caffe {

template <typename Dtype>
void PadChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    num_channels_to_pad_ = this->layer_param_.pad_channel_param().num_channels_to_pad();
    CHECK_GT(num_channels_to_pad_, 0) << "num channels to pad must greater than 0!";
}

template <typename Dtype>
void PadChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] += num_channels_to_pad_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PadChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->height() * bottom[0]->width();
  int channel_by_dim = channels * dim;
  for (int n = 0; n < num; n++){
    caffe_copy(channel_by_dim, bottom_data, top_data);
    bottom_data += channel_by_dim;
    top_data += channel_by_dim;
    caffe_set(num_channels_to_pad_ * dim, Dtype(0), top_data);
    top_data += num_channels_to_pad_ * dim;
  }
}

template <typename Dtype>
void PadChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->height() * bottom[0]->width();
  int channel_by_dim = channels * dim;
  for (int n = 0; n < num; n++){ // just drop the padding derivatives part.
    caffe_copy(channel_by_dim, top_diff, bottom_diff);
    top_diff += (channels + num_channels_to_pad_) * dim;
    bottom_diff += channel_by_dim;
  }
}

#ifdef CPU_ONLY
STUB_GPU(PadChannelLayer);
#endif

INSTANTIATE_CLASS(PadChannelLayer);
REGISTER_LAYER_CLASS(PadChannel);

}  // namespace caffe
