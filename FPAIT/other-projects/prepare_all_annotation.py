# Please use Python 3.x
# python prepare_all_annotation.py xdir_1 xdir_2 ....
# For example : 
# python prepare_all_annotation.py /home/mozi/smokecar/jiaoguan/dec /home/mozi/smokecar/jiaoguan/nov /home/mozi/smokecar/tongzhou/nov
import os, sys, glob, warnings
from os import path as osp
from pathlib import Path
import numpy as np

try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

smoke_cls_names = ('smoke', 'car')


def parse_voc_xml(path):
  xml = ET.parse(path).getroot()
  size_info = xml.find('size')
  foldername = xml.find('folder').text
  filename = xml.find('filename').text
  width  = int(size_info.find('width').text)
  height = int(size_info.find('height').text)
  depth  = int(size_info.find('depth').text)
  objects = []
  classes = []

  find_smoke, crash = False, False
  for obj in xml.iterfind('object'):
    name = obj.find('name').text
    if name not in smoke_cls_names:
      warnings.warn('Unknow class : [{:}], path : {:}'.format(name, path))
      crash = True
      continue
    label = smoke_cls_names.index(name)
    pose = obj.find('pose').text
    truncated = bool(int(obj.find('truncated').text))
    difficult = bool(int(obj.find('difficult').text))
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    objects.append( [xmin, ymin, xmax, ymax] )
    label += 1
    if label == 1: find_smoke = True

    classes.append( label )
  objects = np.array(objects)
  classes = np.array(classes)
  return foldername, filename, objects, classes, [width, height], find_smoke, crash


def load_list_by_voc_format(root_dir):
  anno_dir = osp.join(root_dir, 'annotation')
  pics_dir = osp.join(root_dir, 'raw-dataset')
  assert osp.isdir(anno_dir), 'annotation dir not found : {:}'.format(anno_dir)
  assert osp.isdir(pics_dir), 'annotation dir not found : {:}'.format(pics_dir)
  all_annotations = glob.glob( osp.join(anno_dir, '*', '*.xml') )
  all_annotations.sort()

  ok_smoke, all_oks, oks, datas = 0, 0, 0, []
  empty = 0
  for annotation in all_annotations:
    assert osp.isfile(annotation), 'Not find {}'.format(annotation)
    foldername, filename, objects, classes, image_size, find_smoke, crash = parse_voc_xml( annotation )
    basename = Path(annotation).name.split('.')[0]
    assert basename == filename.split('.')[0], '{:} is wrong, the path is inconsistent with filename [{:}]'.format(annotation, filename)
    
    image_path = osp.join(pics_dir, foldername, filename)
    if not osp.isfile(image_path):
      print ('Not find {:} for {:}'.format(image_path, annotation))
      continue
    else:
      oks = oks + 1
    if find_smoke: ok_smoke = ok_smoke + 1
    if not crash : all_oks  = all_oks + 1
    if objects.shape[0] == 0: empty = empty + 1
  

    cdata = {'image_path':  image_path,
             'annot_path':  annotation,
             'objects':     objects,
             'classes':     classes,
             'crash-label': crash,
             'find_smoke':  find_smoke}
    datas.append( cdata )
  
  print ('The number of annotations : {:}\nCorrect Files: {:} / {:}\nFiles containing smoke-car : {:}\nFiles without objects : {}\nFinally, we collect {} images'.format(len(all_annotations), all_oks, oks, ok_smoke, empty, len(datas)))
  return datas


if __name__ == '__main__':
  assert len(sys.argv) > 1, 'There should at least provide 1 directory'
  xlist = []
  for idx in range(1, len(sys.argv)):
    print ('Processing {:}/{:}-th directory : {:}'.format(idx, len(sys.argv)-1, sys.argv[idx]))
    tlist = load_list_by_voc_format(sys.argv[idx])
    xlist += tlist
  print ('Collect {:} images with annotation'.format( len(xlist) ))
