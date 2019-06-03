# [A Dual-Network Progressive Approach to Weakly Supervised Object Detection](https://d-x-y.github.io/pdf/a-dual-network.pdf)

By Xuanyi Dong, Deyu Meng, Fan Ma, Yi Yang. This paper is accepted by [ACM Multimedia 2017](https://dl.acm.org/citation.cfm?id=3123455).

### Introduction

[**Dual-Network**](https://d-x-y.github.io/pdf/a-dual-network.pdf) is a weakly supervised object detection framework leveraging deep CNN models.

This project is modified on the Matlab code of [R-FCN](https://github.com/daijifeng001/R-FCN) and Fast R-CNN.

### License

Dual-Network is released under the MIT License (refer to the LICENSE file for details).

### Resources & Preparation

0. ImageNet-pretrained networks: [Google Drive](https://drive.google.com/open?id=1xjFL-ZeVzXkY584ZsEnr9O6O3P1Ypjwd). Please save the models into the corresponding sub-directory of `models/pre_trained_models`.
0. The initial pseudo labels for PASCAL VOC 2007 by [ContextLocNet](https://github.com/vadimkantorov/contextlocnet) : [Google Drive](https://drive.google.com/open?id=1wK64uRk6f0DTwUICs3dMy-zZzJEwvZje). Please save and extract it into `data`.
0. The pre-computed region proposals: [Google Drive](https://drive.google.com/open?id=1a0JF0ReqHRbbpkViO_L33GJUIMSf8wV9). Please save and extract it into `data`.
0. Download the PASCAL VOC 2007 data into `datasets`, following the README in `datasets`.
0. Compile `Caffe` located in `external/caffe`.
0. Run `dual_build.m` to complie the nms mex functions.
0. Run `startup.m` to add necessary paths.


### Training & Testing
- [TODO] re-organize the experiment codes.


### Citing Dual-Network

If you find Dual-Network useful in your research, please consider citing:

    @inproceedings{dong2017dual,
        title={A Dual-Network Progressive Approach to Weakly Supervised Object Detection},
        author={Dong, Xuanyi and Meng, Deyu and Ma, Fan and Yang, Yi},
        booktitle={Proceedings of the 2017 ACM on Multimedia Conference},
        pages={279--287},
        year={2017},
        organization={ACM}
    }
    @inproceedings{kantorov2016,
        title = {ContextLocNet: Context-aware Deep Network Models for Weakly Supervised Localization},
        author = {Kantorov, V., Oquab, M., Cho M. and Laptev, I.},
        booktitle = {Proc. European Conference on Computer Vision (ECCV), 2016},
        year = {2016}
    }
    @article{dai16rfcn,
        Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
        Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
        Journal = {arXiv preprint arXiv:1605.06409},
        Year = {2016}
    }
    @inproceedings{girshick2015fast,
        title={Fast R-CNN},
        author={Girshick, Ross},
        booktitle={Proceedings of the IEEE international conference on computer vision},
        pages={1440--1448},
        year={2015}
    }
