#!/usr/bin/env sh
export LD_LIBRARY_PATH=/usr/local/torch/lib:/usr/local/cuda/lib64:/opt/cudnn-8.0-linux-x64-v5.1/lib64
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the depth and the grid-size and the training epoches and model version"
  exit 1
fi

dataset=cifar10
nGPU=4
model=preresnet-acc-$4
epochs=$3
depth=$1
grid=$2

th main.lua -dataset ${dataset} -nGPU ${nGPU} -batchSize 128 -optnet true -netType ${model} -save checkpoints/${dataset}-${model}-${depth}-${epochs}-${grid} -depth ${depth} -nEpochs ${epochs} -grid_size ${grid}
