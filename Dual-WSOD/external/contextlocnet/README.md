# Information & Contact
If you use this code, please cite our work:
> @inproceedings{kantorov2016,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title = {ContextLocNet: Context-aware Deep Network Models for Weakly Supervised Localization},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author = {Kantorov, V., Oquab, M., Cho M. and Laptev, I.},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;booktitle = {Proc. European Conference on Computer Vision (ECCV), 2016},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year = {2016}  
}

The results are available on the [project website](http://www.di.ens.fr/willow/research/contextlocnet) and in the [paper](http://arxiv.org/pdf/1609.04331.pdf) (arXiv [page](http://arxiv.org/abs/1609.04331)). Please submit bugs and ask questions on [GitHub](http://github.com/vadimkantorov/contextlocnet/issues) directly, for other inquiries please contact [Vadim Kantorov](mailto:vadim.kantorov@gmail.com).

This is a joint work of [Vadim Kantorov](http://vadimkantorov.com), [Maxime Oquab](http://github.com/qassemoquab), [Minsu Cho](http://www.di.ens.fr/~mcho), and [Ivan Laptev](http://www.di.ens.fr/~laptev).

# Running the code
1. Install the dependencies: [Torch](http://github.com/torch/distro) with [cuDNN](http://developer.nvidia.com/cudnn) support; [HDF5](http://www.hdfgroup.org/HDF5/); [matio](http://github.com/tbeu/matio); [protobuf](http://github.com/google/protobuf); Luarocks packages [rapidjson](http://github.com/xpol/lua-rapidjson), [hdf5](http://github.com/deepmind/torch-hdf5), [matio](http://github.com/soumith/matio-ffi.torch), [loadcaffe](http://github.com/szagoruyko/loadcaffe), [xml](https://://github.com/lubyk/xml); MATLAB or [octave](https://www.gnu.org/software/octave/) binary in PATH (for computing detection mAP).

  We strongly recommend using [wigwam](http://wigwam.in/) for this (fix the paths to `nvcc` and `libcudnn.so` before running the command):

  ```shell
  wigwam install torch hdf5 matio protobuf octave -DPATH_TO_NVCC="/path/to/cuda/bin/nvcc" -DPATH_TO_CUDNN_SO="/path/to/cudnn/lib64/libcudnn.so"
  wigwam install lua-rapidjson lua-hdf5 lua-matio lua-loadcaffe lua-xml
  wigwam in # execute this to make the installed libraries available
  ```
2. Clone this repository, change the current directory to `contextlocnet`, and compile the ROI pooling module:

  ```shell
  git clone https://github.com/vadimkantorov/contextlocnet
  cd contextlocnet
  (cd ./model && luarocks make)
  ```
3. Download the [VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset and Koen van de Sande's [selective search windows](http://koen.me/research/selectivesearch/) for VOC 2007 and the [VGG-F](https://gist.github.com/ksimonyan/a32c9063ec8e1118221a) model by running the first command. Optionally download the [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and Ross Girshick's [selective search windows](https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_fast_rcnn_models.sh) by manually downloading the [VOC 2012 test data tarball](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar) to `data/common` and then running the second command:
  
  ```shell
  make -f data/common/Makefile download_and_extract_VOC2007 download_VGGF
  # make -f data/common/Makefile download_and_extract_VOC2012
  ```
4. Choose a dataset, preprocess it, and convert the VGG-F model to the Torch format:

  ```shell
  export DATASET=VOC2007
  th preprocess.lua VOC VGGF
  ```
5. Select a GPU and train a model (our best model is `model/contrastive_s.lua`, other choices are `model/contrastive_a.lua`, `model/additive.lua`, and `model/wsddn_repro.lua`):

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  th train.lua model/contrastive_s.lua				# will produce data/model_epoch30.h5 and data/log.json
  ```
6. Test the trained model and compute CorLoc and mAP:

  ```shell
  SUBSET=trainval th test.lua data/model_epoch30.h5 # will produce data/scores_trainval.h5
  th corloc.lua data/scores_trainval.h5			    # will produce data/corloc.json
  SUBSET=test th test.lua data/model_epoch30.h5	    # will produce data/scores_test.h5
  th detection_mAP.lua data/scores_test.h5		    # will produce data/detection_mAP.json
  ```

# Pretrained models for VOC 2007
Model | model_epoch30.h5 | log.json | corloc.json | detection_mAP.json|
:---|:---:|:---:|:---:|:---:|
contrastive_s | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/contrastive_s_model_epoch30.h5) | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/contrastive_s_log.json) | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/contrastive_s_corloc.json) | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/contrastive_s_detection_mAP.json) 
wsddn_repro | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/wsddn_repro_model_epoch30.h5) | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/wsddn_repro_log.json) | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/wsddn_repro_corloc.json) | [link](https://github.com/vadimkantorov/contextlocnet/releases/download/1.0/wsddn_repro_detection_mAP.json)
  
# Acknowledgements & Notes
We greatly thank Hakan Bilen, Relja Arandjelović and Soumith Chintala for fruitful discussion and help.

This work would not have been possible without prior work: Hakan Bilen's [WSDDN](http://github.com/hbilen/WSDDN), Spyros Gidaris's [LocNet](http://github.com/gidariss/LocNet), Sergey Zagoruyko's [loadcaffe](http://github.com/szagoruyko/loadcaffe), Facebook FAIR's [fbnn/Optim.lua](http://github.com/facebook/fbnn/blob/master/fbnn/Optim.lua).

The code is released under the [MIT](http://github.com/vadimkantorov/contextlocnet/blob/master/LICENSE.md) license.
