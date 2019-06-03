import os, sys, pdb
from collections import defaultdict
import torch
import torch.utils.data as data
from copy import deepcopy
from .dataset_utils import pil_loader

class VQADataset(data.Dataset):
  
  def __init__(self, metatask, transform, max_words, vocab_size, train):
    metatask = deepcopy(metatask)
    self.train = train
    self.transform = transform
    self.max_words = max_words
    self.vocab_size = vocab_size

    if self.train:
      dataset = metatask.meta_training
    else:
      dataset = metatask.meta_testing

    self.questions   = []
    self.answers     = []
    self.str_answers = []
    self.images      = []
    self.all_words   = defaultdict(lambda: 0)
    self.words2index = {}

    for data in dataset:
      self.questions.append( data.question )
      self.str_answers.append( data.answer )
      self.images.append( data.image )
      self.all_words[ data.answer ] += 1
    for index, key in enumerate(self.all_words):
      self.words2index[key] = index
    for answer in self.str_answers:
      self.answers.append(self.words2index[answer])
    self.answer_class = len(self.words2index)

  def __len__(self):
    return len(self.images)

  def __repr__(self):
    return ('{name}(train={train}'.format(name=self.__class__.__name__, **self.__dict__) + ', length={:})'.format(len(self)))

  def __getitem__(self, index):
    image = pil_loader(self.images[index])
    if self.transform is not None:
      image = self.transform(image)
    length = len(self.questions[index])
    question = self.questions[index] + [self.vocab_size] * self.max_words
    question = question[:self.max_words]
    answer = self.answers[index]
    
    question = torch.LongTensor(question)
    #length, question, answer = torch.LongTensor([length]), torch.LongTensor(question), torch.LongTensor([answer])
    return image, question, length, answer, torch.LongTensor([index])
