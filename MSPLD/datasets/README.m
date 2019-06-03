# Creat symlinks of VOCdevkit2007 and VOCdevkit2012 here

# Download the training, validation, test data and VOCdevkit for VOC 2007
- wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
- wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
- wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

# VOC2007 train/val/test and VOC2012 train/val is also available here
- link: https://pan.baidu.com/s/1hsxBbiG password: 9vt8

Remember to rename the `VOCdevkit` as `VOCdevkit2007` and `VOCdevkit2012`

# Download the noise data from [YFCC-100M](http://yfcc100m.appspot.com)
We randomly download 18355 images from YFC-100M.
We filter the images by the size limitation of [300, 800] and use 10000 images of them as the noise distractors for PASCAL VOC 2007. These images can be found in [Google Drive](https://drive.google.com/open?id=1J1oOlTZoXa0HzL6CUnsbxMTpGusz_rAx).
