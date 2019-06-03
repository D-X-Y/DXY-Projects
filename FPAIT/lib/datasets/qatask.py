import numpy as np
import os, pdb, random, torch
from pathlib import Path
from logger import print_log
from .coco_utils import TQASample, SimpleQA

class MetaQATask(object):

  def __init__(self, dataset, pth_data, num_classes, num_samples_per_class, log):
    """
    Args:
      num_samples_per_class: num samples to generate per class in one batch
      batch_size: size of meta batch size (e.g. number of functions)
    """
    self.dataset = dataset.lower()
    self.num_samples_per_class = num_samples_per_class
    self.num_classes = num_classes
    self.pth_data = Path(pth_data).resolve()
    assert self.pth_data.exists(), 'The {} does not exists.'.format(self.pth_data)
  
    if self.dataset == 'toronto-coco-qa':
      data = torch.load(self.pth_data)
      self.all_words     = data['all_words']
      self.words_index   = data['words_index']
      self.max_word      = len(self.words_index)
      for key in self.all_words:
        self.words_index[ int(self.words_index[key]) ] = key
      self.all_answers   = data['all_answers']
      #self.answers_index = data['answers_index']
      self.base_training      = data['training']
      self.base_testing       = data['testing']
      
      self.meta_training      = [SimpleQA(self.words_index, x) for x in self.base_training]
      self.meta_testing       = [SimpleQA(self.words_index, x) for x in self.base_testing]
      for x in self.meta_training + self.meta_testing:
        x.check(self.max_word)

    elif self.dataset == 'ms-coco-ic':
      data = torch.load(self.pth_data)
      self.all_words     = data['all_words']
      self.words_index   = data['words2index']
      self.max_word      = len(self.words_index)
      for key in self.all_words:
        self.words_index[ int(self.words_index[key]) ] = key
      self.base_training      = data['training']
      self.base_testing       = data['testing']

      self.meta_training      = [SimpleQA(self.words_index, x, False) for x in self.base_training]
      self.meta_testing       = [SimpleQA(self.words_index, x, False) for x in self.base_testing]
      for x in self.meta_training + self.meta_testing:
        x.check(self.max_word)
    else:
      raise ValueError('Unrecognized dataset : {}'.format(dataset))
    print_log('Meta-Task : [{:}], train={:}, test={:}'.format(dataset, len(self.meta_training), len(self.meta_testing)), log)

  def __repr__(self):
    return ('{name}(dataset={dataset}, {num_classes}-way {num_samples_per_class}-shot, pth={pth_data})'.format(name=self.__class__.__name__, **self.__dict__))
