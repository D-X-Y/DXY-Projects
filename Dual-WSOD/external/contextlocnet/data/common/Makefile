DOWNLOADS = ./data/common

download_and_extract_VOC2007:
	wget -P $(DOWNLOADS) \
		http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar \
		http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar \
		http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar \
		http://koen.me/research/downloads/SelectiveSearchVOC2007trainval.mat \
		http://koen.me/research/downloads/SelectiveSearchVOC2007test.mat
	cd $(DOWNLOADS) && for f in VOCdevkit_08-Jun-2007.tar VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar; do tar -xf $$f; done && mv VOCdevkit VOCdevkit_2007
  
download_and_extract_VOC2012:
	wget -P $(DOWNLOADS) \
		http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar \
		http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar \
		http://people.eecs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
	cd $(DOWNLOADS) && for f in VOCdevkit_18-May-2011.tar VOCtrainval_11-May-2012.tar VOC2012test.tar selective_search_data.tgz; do tar -xf $$f; done && mv VOCdevkit VOCdevkit_2012

download_VGGF:
	wget -P $(DOWNLOADS) \
		http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_F.caffemodel \
		https://gist.githubusercontent.com/ksimonyan/a32c9063ec8e1118221a/raw/6a3b8af023bae65669a4ceccd7331a5e7767aa4e/VGG_CNN_F_deploy.prototxt
  
.PHONY: download_and_extract_VOC2007 download_and_extract_VOC2012 download_VGGF
