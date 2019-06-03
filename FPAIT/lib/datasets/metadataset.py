import torch
import torch.utils.data as data
from copy import deepcopy
from .dataset_utils import pil_loader

class MetaDataset(data.Dataset):
   
  def __init__(self, metatask, transform, train):
    self.metatask = deepcopy(metatask)
    self.length = len(self.metatask)
    self.train = train
    self.transform = transform
  
  def set_length(self, length):
    if length < 0: self.length = len(self.metatask)
    else         : self.length = length
  
  def __len__(self):
    return self.length

  def __repr__(self):
    return ('{name}(meta={metatask}, train={train}, length={length})'.format(name=self.__class__.__name__, **self.__dict__))

  def __getitem__(self, idx):
    #imagesA, classesA, imagesB, classesB = self.metatask.random_sample(self.train)
    datasA, datasB = self.metatask.random_sample(self.train)
    tensorA, tensorB, classesA, classesB = [], [], [], []

    for (image, cls) in datasA:
      image = pil_loader(image)
      if self.transform is not None:
        image = self.transform(image)
      tensorA.append(image)
      classesA.append(cls)
    for (image, cls) in datasB:
      image = pil_loader(image)
      if self.transform is not None:
        image = self.transform(image)
      tensorB.append(image)
      classesB.append(cls)
    tensorA, tensorB = torch.stack(tensorA), torch.stack(tensorB)
    clssesA, clssesB = torch.LongTensor(classesA), torch.LongTensor(classesB)
    return tensorA, clssesA, tensorB, clssesB
