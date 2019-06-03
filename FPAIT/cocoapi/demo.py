from pycocotools.coco import COCO
import pdb, numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pathlib import Path
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

MSCOCO_ROOT = '/Users/xuanyidong/Downloads/MS-COCO'
annotation_file = '{}/annotations/captions_train2014.json'.format(MSCOCO_ROOT)

def load_caption(root, subset):
  root = Path(root)
  coco_ann_file = root / 'annotations' / 'instances_{}.json'.format(subset)
  coco = COCO(coco_ann_file)
  coco_caps_file = root / 'annotations' / 'captions_{}.json'.format(subset)
  coco_caps = COCO(coco_caps_file)

  # display COCO categories and supercategories
  cats = coco.loadCats(coco.getCatIds())
  names = [cat['name'] for cat in cats]
  print ('COCO categories [{:}] : \n{:}\n'.format(len(names), ' '.join(names)))

  supernames = set([cat['supercategory'] for cat in cats])
  print('COCO supercategories [{:}]: \n{:}'.format(len(supernames), ' '.join(supernames)))

  # get all images containing given categories, select one at random
  catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
  imgIds = coco.getImgIds(catIds=catIds)
  imgIds = coco.getImgIds(imgIds = [23004])
  img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
  # load and display image
  # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
  # use url to load image
  I = io.imread(img['coco_url'])
  # load and display caption annotations
  annIds = coco_caps.getAnnIds(imgIds=img['id']);
  anns = coco_caps.loadAnns(annIds)
  coco_caps.showAnns(anns)
  plt.imshow(I); plt.axis('off'); plt.show()
  
  pdb.set_trace()
  #imgIds = coco.getImgIds(imgIds = [324158])
  #img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

if __name__ == '__main__':
  load_caption(MSCOCO_ROOT, 'train2014')
