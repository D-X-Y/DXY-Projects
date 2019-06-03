# python show-data.py
# show the data of TieredImageNet
import cv2, numpy as np
from os import path as osp
from tqdm import tqdm
import pickle as pkl


def decompress(path, output):
  with open(output, 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    array = u.load()
    #array = pkl.load(f)
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  print ('There are {:} images'.format( len(array) ))
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)


def read_cache(tiered_image_dir, mode):
  """Reads dataset from cached pkl file."""
  cache_path_images = osp.join(tiered_image_dir, '{:}_images.npz'.format(mode))
  cache_path_labels = osp.join(tiered_image_dir, '{:}_labels.pkl'.format(mode))

  # Decompress images.
  if not osp.exists(cache_path_images):
    png_pkl = cache_path_images[:-4] + '_png.pkl'
    assert osp.exists(png_pkl), '{:} does not exist'.format(png_pkl)
    decompress(cache_path_images, png_pkl)
  assert osp.exists(cache_path_labels) and osp.exists(cache_path_images)
  try:
    with open(cache_path_labels, "rb") as f:
      data = pkl.load(f, encoding='bytes')
      _label_specific = data[b"label_specific"]
      _label_general = data[b"label_general"]
      _label_specific_str = data[b"label_specific_str"]
      _label_general_str = data[b"label_general_str"]
  except:
    with open(cache_path_labels, "rb") as f:
      data = pkl.load(f)
      _label_specific = data["label_specific"]
      _label_general = data["label_general"]
      _label_specific_str = data["label_specific_str"]
      _label_general_str = data["label_general_str"]
  with np.load(cache_path_images, mmap_mode="r", encoding='latin1') as data:
    _images = data["images"]
  print ('images : {:}'.format(_images.shape))
  print ('label_specific : {:}, max={:}'.format(_label_specific.shape, _label_specific.max()))
  print ('label_general  : {:}, max={:}'.format(_label_general.shape , _label_general.max()))

if __name__ == '__main__':
  read_cache('.', 'val')
