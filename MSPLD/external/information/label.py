import pdb, os, sys
import numpy as np
import os.path as osp
from collections import defaultdict
import cv2
import glob

Font = cv2.FONT_HERSHEY_SIMPLEX

class ImageNet(object):
  def __init__(self, name, cls, description):
    self.name = name
    self.cls  = cls
    self.description = description
    self.det  = None

  def convert(self):
    string = '{:9s} {:4d} {:30s}'.format(self.name, self.cls, self.description)
    if self.det is not None:
      string = string + ' {:10s}'.format(self.det)
    return string
  
  def load(self, string):
    alls = string.strip().split(' ')
    alls = [x for x in alls if x is not '']
    assert len(alls)>=3 and len(alls)<=4
    self.name = alls[0]
    self.cls  = int(alls[1])
    self.description = alls[2]
    if len(alls) > 3:
      self.det = alls[3]

def load_txt(path):
  cfile = open(path, 'r')
  objects = []
  for line in cfile.readlines():
    x = ImageNet(None, None, None)
    x.load(line)
    objects.append(x)
  cfile.close()
  return objects
 
def load_meta(txt, N):
  cfile = open(txt, 'r')
  meta  = defaultdict(list)
  num = 0
  for line in cfile.readlines():
    line = line.strip()
    line = line.split(':')
    assert len(line) == 2, '{:} : [{:}]'.format(txt, line)
    line[0] = line[0].replace(' ', '')
    objs = line[1].split(', ')
    objs = [x.replace(' ', '').lower() for x in objs]
    meta[line[0]] = objs
    num += len(objs)
  cfile.close()
  assert num == N, '{:} vs {:}'.format(num, N)
  return meta

def print_meta(meta):
  for key, value in meta.items():
    print ('[{:10s}] : {:}'.format(key, value))

def save_txt(objects, out_file, Clear=False):
  cfile = open(out_file, 'w')
  for obj in objects:
    if Clear: obj.det = None
    cfile.write('{:}\n'.format(obj.convert()))
  cfile.close()

def find_obj(inputs, meta):
  if inputs.lower() == 'none': return True, None, 'None'
  inputs = inputs.lower()
  inputs = inputs.split('.')
  ssssss = []
  _value = None
  for key, value in meta.items():
    if key.lower().find(inputs[0]) == 0:
      ssssss.append(key)
      _value = value
  if len(ssssss) == 0: return False, None, None
  if len(ssssss) >  1: return False, '{:}'.format(ssssss), None
  if len(inputs) >  2 or len(inputs) == 1: return False, '{:}'.format(ssssss), None

  A = []
  for x in _value:
    if x.lower().find(inputs[1]) == 0:
      A.append(x)

  if len(A) == 0: return False, '{:}'.format(ssssss), None
  if len(A) >  1: return False, '{:}'.format(ssssss), '{:}'.format(A)
  assert len(A) == 1

  return True, ssssss[0], A[0]


def draw_text(img, inputs, meta, message=None):

  color = (255, 255, 255)
  offset, gap = 30, 40
  cv2.putText(img, inputs, (10, offset), Font, 1, color, 1)
  offset += gap

  for key, value in meta.items():
    string = '[{:10s}] : {:}'.format(key, value)
    cv2.putText(img, string, (10, offset), Font, 1, color, 2)
    offset += gap

  cv2.putText(img, '{:}'.format(message), (10, offset), Font, 1, color, 2)
 
def generate(images, side):
  xs = []
  half = side // 2
  for i in range(4):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (half,half), interpolation=cv2.INTER_CUBIC)
    xs.append(img)
  A = np.concatenate((xs[0], xs[1]), axis=0)
  B = np.concatenate((xs[2], xs[3]), axis=0)
  img = np.concatenate((A, B), axis=1)
  img = cv2.resize(img, (side,side), interpolation=cv2.INTER_CUBIC)
  return img
  

def main(in_file, out_file, idir, meta):
  objects = load_txt(in_file)
  for obj in objects:
    cdir = osp.join(idir, obj.name)
    assert osp.isdir(cdir), 'Sub-dir {:} not find'.format(cdir)

  #save_txt(objects, 'ori-imagenet-1k', True)
  side, resize = 1500, 800

  less = sum([1 for obj in objects if obj.det is None])

  for obj in objects:
    if obj.det is not None: continue
    print ('{:} {:} {:}   still needs {:}'.format(obj.name, obj.cls, obj.description, less))
    cdir = osp.join(idir, obj.name)
    images = glob.glob( osp.join(cdir, '*.JPEG') )
    image = images[0]
    inputs = ''
    message = None

    while True:
      img = generate(images, side)
      draw_text(img, inputs, meta, message)
      img = cv2.resize(img, (resize,resize), interpolation=cv2.INTER_CUBIC)
      cv2.imshow('image', img)
      key = cv2.waitKey(0)
      if key == 27:         # ESC
        message = None
        break
      elif key == 8:        # delete
        inputs = inputs[:-1] if len(inputs) > 0 else inputs
      elif key == 13:       # enter-return
        ok, superx, basex = find_obj(inputs, meta)
        if ok:
          obj.det = basex
          message = '[OK!] {:} >>> {:}'.format(superx, basex)
        else:
          message = '[CAO] {:} >>> {:}'.format(superx, basex)

        print ('Save {:}, {:} into {:} with message = {:}'.format(obj.name, obj.cls, out_file, message))
        save_txt(objects, out_file)

        if ok:
          message = None
          less = less - 1
          break
      elif (key >= ord('a') and key <= ord('z')) or (key == ord('.')) or (key == ord('-')):
        inputs += chr(key)
      else:
        print ('Unkown key : {:}'.format(key))

  cv2.destroyAllWindows()

  none_num = 0
  for obj in objects:
    if obj.det == 'None': none_num += 1
  print ('total : {:}, none : {:}'.format(len(objects), none_num))
   

if __name__ == '__main__':
  imagenet_dir = '/Users/dongxuanyi/datasets/ILSVRC2012/val'
  coco = load_meta('ms-coco',   80)
  pvoc = load_meta('pascal_voc',20)
  print ('MS-COCO')
  print_meta (coco)
  print ('PASCAL-VOC')
  print_meta (pvoc)


  #in_file   = 'ImageNet-1000.pvoc'
  #pvoc_file = 'ImageNet-1000.pvoc'
  #main(in_file, pvoc_file, imagenet_dir, pvoc)

  coco_file = 'ImageNet-1000.coco'
  main(coco_file, coco_file, imagenet_dir, coco)
