# Models used in this paper

## Real-world Experiment Setup

### CIFAR-10

We use 14 CNN models to extract features.
11 models are [ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110, Pre-ResNet-20, Pre-ResNet-32, Pre-ResNet-44, Pre-ResNet-56, Pre-ResNet-110, Pre-ResNet-164](https://github.com/D-X-Y/ResNeXt-DenseNet).
One model is [CIFAR-10-Full](https://github.com/D-X-Y/HCMF/blob/master/support/cifar10_full.proto).
Two models are the modified GoogleNet and VGG-16. The strides in conv4 and conv5 are changed from 2 to 1, and fc layers are changed to global pooling layers.

### CIFAR-100

We use the last pooling layer features from ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-68, ResNet-110 and the Pre-ResNet-20, Pre-ResNet-32, Pre-ResNet-44, Pre-ResNet-56, Pre-ResNet-110, and Pre-ResNet-164.

### UCF-101

We use five different features, i.e., ``fc6'' of C3D, ``pool5/7x7\_s1'' from GoogleNet, ``pool5'' of ResNet-152, and two ``fc6''s of Two-Stream Networks.

### Oxford-IIIT-Pet & PASCAL VOC 2007

We extract eight different kinds of features to train SVM classifiers.
Four features of them are from the ``fc6'' layer of AlexNet, VGG16/19, CaffeNet.
One feature of them is from the ``pool5/7x7\_s1'' layer of GoogleNet.
Three features of them are from the ``pool5'' layer of the ResNet-50, ResNet-101, ResNet-152 models.
