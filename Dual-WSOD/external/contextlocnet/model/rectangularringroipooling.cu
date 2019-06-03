//copied from https://github.com/gidariss/caffe_LocNet/blob/d2ba49552068958556b98ba382610ea865add17c/src/caffe/layers/region_pooling_layer.cu

#include "luaT.h"
#include "THC.h"

#include <lua.h>
#include "THCGeneral.h"

#define CAFFE_CUDA_NUM_THREADS 1024

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
	if(error != cudaSuccess) { printf("CUDA ERROR. %s\n", cudaGetErrorString(error)); }; \
  } while (0)

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, ph, pw) is an element in the pooled output
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		// For each ROI R = [batch_index, x_outer_1, y_outer_1, x_outer_2, y_outer_2, x_inner_1, y_inner_1, x_inner_2, y_inner_2]: 
		// where R_outer = [x_outer_1, y_outer_1, x_outer_2, y_outer_2] is the outer rectangle of the region and 
		// R_inner = [x_inner_1, y_inner_1, x_inner_2, y_inner_2] is the inner rectangle of the region
		// max pooler over R by ignoring (setting to zero) the activations that lay inside the inner rectangle R_inner

		bottom_rois += n * 9;
		int roi_batch_ind = bottom_rois[0];


		// outer rectangle of the region
		int roi_start_w   = int(bottom_rois[1] );//* spatial_scale);
		int roi_start_h   = int(bottom_rois[2] );//* spatial_scale);
		int roi_end_w     = int(bottom_rois[3] );//* spatial_scale);
		int roi_end_h     = int(bottom_rois[4] );//* spatial_scale);

		// inner rectangle of the region
		int roi_start_w_in = int(bottom_rois[5]);//* spatial_scale);
		int roi_start_h_in = int(bottom_rois[6]);//* spatial_scale);
		int roi_end_w_in   = int(bottom_rois[7]);//* spatial_scale);
		int roi_end_h_in   = int(bottom_rois[8]);//* spatial_scale);

		// Force malformed ROIs to be 1x1
		int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
		int roi_height = max(roi_end_h - roi_start_h + 1, 1);
		Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
		Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);

		const int hstart = min(height, max(0, static_cast<int>(floor(static_cast<Dtype>(ph)   * bin_size_h)) + roi_start_h));
		const int hend   = min(height, max(0, static_cast<int>(ceil( static_cast<Dtype>(ph+1) * bin_size_h)) + roi_start_h));
		const int wstart = min(width,  max(0, static_cast<int>(floor(static_cast<Dtype>(pw)   * bin_size_w)) + roi_start_w));
		const int wend   = min(width,  max(0, static_cast<int>(ceil( static_cast<Dtype>(pw+1) * bin_size_w)) + roi_start_w));

		Dtype maxval = 0; 

		int maxidx = -1;
		bottom_data += (roi_batch_ind * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (!(w > roi_start_w_in && w < roi_end_w_in && h > roi_start_h_in && h < roi_end_h_in)) {
					// if it is not inside the inner rectangle of the region
					int bottom_index = h * width + w;
					if (bottom_data[bottom_index] > maxval) {
						maxval = bottom_data[bottom_index];
						maxidx = bottom_index;
					}
				}
			}
		}
		top_data[index] = maxval;
		argmax_data[index] = maxidx;
	}
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;

		Dtype gradient = 0;
		// Accumulate gradient over all ROIs that pooled this element
		for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
			const Dtype* offset_bottom_rois = bottom_rois + roi_n * 9;
			int roi_batch_ind = offset_bottom_rois[0];
			// Skip if ROI's batch index doesn't match n
			if (n != roi_batch_ind) {
				continue;
			}


			// outer rectangle of the region
			int roi_start_w   = int(offset_bottom_rois[1]);// * spatial_scale);
			int roi_start_h   = int(offset_bottom_rois[2]);// * spatial_scale);
			int roi_end_w     = int(offset_bottom_rois[3]);// * spatial_scale);
			int roi_end_h     = int(offset_bottom_rois[4]);// * spatial_scale);

			// inner rectangle of the region
			int roi_start_w_in= int(offset_bottom_rois[5]);// * spatial_scale);
			int roi_start_h_in= int(offset_bottom_rois[6]);// * spatial_scale);
			int roi_end_w_in  = int(offset_bottom_rois[7]);// * spatial_scale);
			int roi_end_h_in  = int(offset_bottom_rois[8]);// * spatial_scale);


			// Skip if ROI doesn't include (h, w)
			const bool in_roi =  (w >= roi_start_w && w <= roi_end_w &&
					h >= roi_start_h && h <= roi_end_h) && 
				!(w > roi_start_w_in && w < roi_end_w_in && 
						h > roi_start_h_in && h < roi_end_h_in);

			if (!in_roi) {
				continue;
			}

			int top_offset = (roi_n * channels + c) * pooled_height * pooled_width;
			const Dtype* offset_top_diff = top_diff + top_offset;
			const int* offset_argmax_data = argmax_data + top_offset;

			// Compute feasible set of pooled units that could have pooled
			// this bottom unit

			// Force malformed ROIs to be 1x1
			int roi_width = max(roi_end_w - roi_start_w + 1, 1);
			int roi_height = max(roi_end_h - roi_start_h + 1, 1);

			Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);

			int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
			int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
			int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
			int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

			phstart = min(max(phstart, 0), pooled_height);
			phend = min(max(phend, 0), pooled_height);
			pwstart = min(max(pwstart, 0), pooled_width);
			pwend = min(max(pwend, 0), pooled_width);

			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
						gradient += offset_top_diff[ph * pooled_width + pw];
					}
				}
			}
		}
		bottom_diff[index] = gradient;
	}
}

THCState* getCutorchState(lua_State* L)
{
    lua_getglobal(L, "cutorch");
    lua_getfield(L, -1, "getState");
    lua_call(L, 0, 1);
    THCState *state = (THCState*) lua_touserdata(L, -1);
    lua_pop(L, 2);
    return state;
}

static int updateOutput(lua_State *L)
{
	THCState *state = getCutorchState(L);
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *rois = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
	THCudaIntTensor *argmax = (THCudaIntTensor *)luaT_getfieldcheckudata(L, 1, "argmax", "torch.CudaIntTensor");

	int pooled_height_ = luaT_getfieldcheckint(L, 1, "pooled_height");
	int pooled_width_ = luaT_getfieldcheckint(L, 1, "pooled_width");
	THCudaTensor_resize5d(state, output, THCudaTensor_size(state, rois, 0), THCudaTensor_size(state, rois, 1), THCudaTensor_size(state, input, 1), pooled_height_, pooled_width_);
	THCudaIntTensor_resize5d(state, argmax, THCudaTensor_size(state, rois, 0), THCudaTensor_size(state, rois, 1), THCudaTensor_size(state, input, 1), pooled_height_, pooled_width_);

	const float* bottom_data = THCudaTensor_data(state, input);
	const float* bottom_rois = THCudaTensor_data(state, rois);
	float* top_data = THCudaTensor_data(state, output);
	int* argmax_data = THCudaIntTensor_data(state, argmax); // int -> float
	
	// TODO: BATCH
	// BDHW 1DHW
	int count = THCudaTensor_nElement(state, output); // top[0]->count();
	int channels_ = THCudaTensor_size(state, input, 1);
	int height_ = THCudaTensor_size(state, input, 2);
	int width_ = THCudaTensor_size(state, input, 3);
	float spatial_scale_ = luaT_getfieldchecknumber(L, 1, "spatial_scale");

	CUDA_POST_KERNEL_CHECK;
	
	// NOLINT_NEXT_LINE(whitespace/operators)
	ROIPoolForward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			  count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
			  pooled_width_, bottom_rois, top_data, argmax_data);
	CUDA_POST_KERNEL_CHECK;

	return 1;
}

static int updateGradInput(lua_State *L)
{
	THCState *state = getCutorchState(L);
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *rois = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaIntTensor *argmax = (THCudaIntTensor *)luaT_getfieldcheckudata(L, 1, "argmax", "torch.CudaIntTensor");

	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

	THCudaTensor_resizeAs(state, gradInput, input);
	THCudaTensor_zero(state, gradInput);
	
	const float* bottom_rois = THCudaTensor_data(state, rois);
	const float* top_diff = THCudaTensor_data(state, gradOutput);
	float* bottom_diff = THCudaTensor_data(state, gradInput);
	int* argmax_data = THCudaIntTensor_data(state, argmax);

	const int count = THCudaTensor_nElement(state, gradInput);
    int channels_ = THCudaTensor_size(state, input, 1);
	int height_ = THCudaTensor_size(state, input, 2);
	int width_ = THCudaTensor_size(state, input, 3);
	int pooled_height_ = luaT_getfieldcheckint(L, 1, "pooled_height");
	int pooled_width_ = luaT_getfieldcheckint(L, 1, "pooled_width");
	float spatial_scale_ = luaT_getfieldchecknumber(L, 1, "spatial_scale");
	int num_rois = THCudaTensor_size(state, rois, 0) * THCudaTensor_size(state, rois, 1); // bachSize x numRoisPerImage

	// NOLINT_NEXT_LINE(whitespace/operators)
	CUDA_POST_KERNEL_CHECK;
	ROIPoolBackward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, top_diff, argmax_data, num_rois, spatial_scale_, channels_,
			height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
	CUDA_POST_KERNEL_CHECK;

	return 1;
}

static const struct luaL_Reg lua_registrations [] = {
  {"updateOutput", updateOutput},
  {"updateGradInput", updateGradInput},
  {NULL, NULL}
};

LUA_EXTERNC DLL_EXPORT int luaopen_libcucontextlocnet(lua_State *L)
{
  lua_newtable(L);

  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, lua_registrations, "contextlocnet");
  lua_pop(L,1);

  return 1;
}
