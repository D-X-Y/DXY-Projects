#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

CAFFE=build/tools/caffe 

time GLOG_log_dir=examples/FRCNN/log $CAFFE train   \
    --gpu $gpu \
    --solver models/FRCNN/vgg16/solver.proto \
    --weights models/FRCNN/VGG16.v2.caffemodel

time python examples/FRCNN/convert_model.py \
    --model models/FRCNN/vgg16/test.proto \
    --weights models/FRCNN/snapshot/vgg16_faster_rcnn_iter_70000.caffemodel \
    --config examples/FRCNN/config/voc_config.json \
    --net_out models/FRCNN/vgg16_faster_rcnn_final.caffemodel
