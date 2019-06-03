#include "caffe/layers/pad_channel_layer.hpp"

namespace caffe {

    // Copy (one line per thread) from one array to another, with arbitrary
    // strides in the last two dimensions.
    template <typename Dtype>
    __global__ void pad_forward_kernel(const int dst_count, const int src_channels, const int dst_channels,
        const int dim,  const Dtype* src, Dtype* dst) 
    {
        CUDA_KERNEL_LOOP(index, dst_count)
        {
            int num = index / (dim * dst_channels);
            int dst_c = index / dim % dst_channels;
            int pixel_pos = index % dim;
            if (dst_c < src_channels)
                dst[index] = src[num * src_channels * dim + dst_c * dim + pixel_pos];
            else
                dst[index] = Dtype(0);
        }
    }


    template <typename Dtype>
    void PadChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();

        int src_channels = bottom[0]->channels();
        int dim = bottom[0]->height() * bottom[0]->width();
        int dst_channels = src_channels + num_channels_to_pad_;
        const int dst_count = top[0]->count();
        pad_forward_kernel<Dtype> << <CAFFE_GET_BLOCKS(dst_count), CAFFE_CUDA_NUM_THREADS >> >(
            dst_count, src_channels, dst_channels, dim, bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }

    template <typename Dtype>
    __global__ void pad_backward_kernel(const int bottom_count, const int bottom_channels, const int top_channels,
        const int dim, const Dtype* top, Dtype* bottom)
    {
        CUDA_KERNEL_LOOP(index, bottom_count)
        {
            int num = index / (dim * bottom_channels);
            int bottom_c = index / dim % bottom_channels;
            int pixel_pos = index % dim;
            bottom[index] = top[num * top_channels * dim + bottom_c * dim + pixel_pos];
        }
    }

    template <typename Dtype>
    void PadChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        int bottom_count = bottom[0]->count();
        int bottom_channels = bottom[0]->channels();
        int dim = bottom[0]->height() * bottom[0]->width();
        int top_channels = bottom_channels + num_channels_to_pad_;
        pad_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS >> >(
            bottom_count, bottom_channels, top_channels, dim, top_diff, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
    }

    INSTANTIATE_LAYER_GPU_FUNCS(PadChannelLayer);

}  // namespace caffe
