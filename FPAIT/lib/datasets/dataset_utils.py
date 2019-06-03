from PIL import Image
import numpy as np
import copy, math

def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    with Image.open(f) as img:
      return img.convert('RGB')
