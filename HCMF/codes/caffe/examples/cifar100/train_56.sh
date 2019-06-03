if [ ! -n "$1" ] ;then
	echo "\$1 is empty, default is 0"
	gpu=0
else
	echo "use $1-th gpu"
	gpu=$1
fi
GLOG_log_dir=examples/cifar100/log ./build/tools/caffe train --solver examples/cifar100/solver_56.proto --gpu $gpu
