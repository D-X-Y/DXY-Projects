# Pre-trained Data

The full ImageNet dataset contains the classes that appear in the PASCAL VOC and MS COCO. We would like to see what if these detection classes do not show in the pre-trained data
.
For this purpose, we manually label whether one ImageNet class is the same as one detection class or not (20 classes in PASCAL VOC and 80 classes in MS COCO).

## PASCAL VOC
See `ImageNet-1000.pvoc`.
Each line is in the following format.
```
image-label index description detection-class
```
If the `detection-class` is `None`, then this class is not a detection class.

## MS COCO
See `ImageNet-1000.coco`.
Each line is in the following format.
```
image-label index description detection-class
```
If the `detection-class` is `None`, then this class is not a detection class.
