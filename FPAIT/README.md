# Fast Parameter Adaptation for Few-shot Image Captioning and Visual Question Answering

By Xuanyi Dong, Linchao Zhu, De Zhang, Yi Yang, Fei Wu

## Data Preparation

### Download
Make directory at `~/datasets/MS-COCO`.
- download the ms-coco [train](http://images.cocodataset.org/zips/train2014.zip), [val](http://images.cocodataset.org/zips/val2014.zip), and [test](http://images.cocodataset.org/zips/test2014.zip).
- download the [trainval2014-annotation](http://images.cocodataset.org/annotations/annotations_trainval2014.zip), and the [test info](http://images.cocodataset.org/annotations/image_info_test2014.zip).
- organize the data as follows, where trainval2014 contains all the trainval images and test2014 contains all the test images.
- download [cap_coco.json](https://drive.google.com/open?id=1Po2yaZEMplI2_oSZRhMv6uCc1v4Tp0OS) into `./data/COCO-Caption/`.
```
- ~/datasets/MS-COCO
-- annotations
--- captions_train2014.json captions_val2014.json image_info_test2014.json instances_train2014.json instances_val2014.json
-- test2014
-- trainval2014
```


### Compile coco api
```
cd cocoapi
make
```


### Few-shot Image Caption
In the directory `data`, run:
```
python Generate_Caption.py
```
After run the above command, you can obtain `data/COCO-Caption/few-shot-coco.pth` for few-shot image caption.


### Few-shot Visual Question Answering
In the directory `data`, run:
```
python Generate_VQA.py
```
After run the above command, you can obtain `data/Toronto-COCO-QA/object.pth` for few-shot visual question answering.

### Show Samples
We give an example to show how to read the pre-processed data
```
python show_data.py
```


## Citation
If you find this project help your research, please cite:
```
@inproceedings{dong2018fpait,
  title     = {Fast Parameter Adaptation for Few-shot Image Captioning and Visual Question Answering},
  author    = {Dong, Xuanyi and Zhu, Linchao and Zhang, De and Yang, Yi and Wu, Fei},
  booktitle = {Proceedings of the 2018 ACM on Multimedia Conference},
  year      = {2018}
}
```
