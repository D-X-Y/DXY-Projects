import os, sys, random, pdb
import numpy as np
from collections import defaultdict
import torch
import torch.utils.data as data
from copy import deepcopy
from .dataset_utils import pil_loader

class VQAMetaDataset(data.Dataset):
  
  def __init__(self, metatask, transform, max_words, vocab_size, train):
    metatask = deepcopy(metatask)
    self.question_words2index = metatask.words_index
    # meta setting
    self.n_way = metatask.num_classes
    self.k_shot = metatask.num_samples_per_class
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

    # regards all meta-classes
    self.class_search = [[] for _ in range(self.answer_class)]
    for idx, (question, answer, str_answer, image) in enumerate(zip(self.questions, self.answers, self.str_answers, self.images)):
      self.class_search[answer].append( idx )
  
    for iclass in range(self.answer_class):
      assert len(self.class_search[iclass]) >= 2 * self.k_shot, 'The [{:}]-th class is {:}'.format(iclass, self.class_search[iclass])
    self.all_classes = [x for x in range(self.answer_class)]
    self.dataset_length = len(self.images)

  def obtain_string(self, index):
    image = self.images[index]
    question = self.questions[index]
    question = [self.question_words2index[x] for x in question]
    question, answer = ' '.join(question), self.str_answers[index]
    return str(image), question, answer

  def set_length(self, x):
    self.dataset_length = int(x)

  def __len__(self):
    return self.dataset_length

  def __repr__(self):
    return ('{name}(train={train}, {n_way}-way {k_shot}-shot'.format(name=self.__class__.__name__, **self.__dict__) + ', length={:})'.format(len(self)))

  def __getitem__(self, index):
    # randomly select one meta-sample
    classes = random.sample(self.all_classes, self.n_way)
    base_train_data, base_test_data = [], []
    for i, cls in enumerate(classes):
      selected_index = random.sample(self.class_search[cls], self.k_shot*2)
      train_index, test_index = selected_index[:self.k_shot], selected_index[self.k_shot:]
      #base_train_data += [(self.questions[x], self.answers[x], self.images[x]) for x in train_index]
      #base_test_data  += [(self.questions[x], self.answers[x], self.images[x]) for x in test_index]
      base_train_data += [(x, i) for x in train_index]
      base_test_data  += [(x, i) for x in test_index]
    random.shuffle(base_train_data)
    random.shuffle(base_test_data)

    training_images, training_questions, training_lengths, training_answers, training_indexes = [], [], [], [], []
    for index, cur_cls in base_train_data:
      image, question, length, answer, index = self.getitem(index)
      training_images.append(image)
      training_questions.append(question)
      training_lengths.append(length)
      training_answers.append(cur_cls)
      training_indexes.append(index)
    training_images = torch.stack(training_images)
    training_questions = torch.stack(training_questions)
    training_lengths = torch.LongTensor(training_lengths)
    training_answers = torch.LongTensor(training_answers)
    training_indexes = torch.LongTensor(training_indexes)

    testing_images, testing_questions, testing_lengths, testing_answers, testing_indexes = [], [], [], [], []
    for index, cur_cls in base_test_data:
      image, question, length, answer, index = self.getitem(index)
      testing_images.append(image)
      testing_questions.append(question)
      testing_lengths.append(length)
      testing_answers.append(cur_cls)
      testing_indexes.append(index)
    testing_images  = torch.stack(testing_images)
    testing_questions = torch.stack(testing_questions)
    testing_lengths = torch.LongTensor(testing_lengths)
    testing_answers = torch.LongTensor(testing_answers)
    testing_indexes = torch.LongTensor(testing_indexes)
    return training_images, training_questions, training_lengths, training_answers, training_indexes, testing_images, testing_questions, testing_lengths, testing_answers, testing_indexes

  def getitem(self, index):
    image = pil_loader(self.images[index])
    if self.transform is not None:
      image = self.transform(image)
    length = len(self.questions[index])
    question = self.questions[index] + [self.vocab_size] * self.max_words
    question = question[:self.max_words]
    answer = self.answers[index]
    
    question = torch.LongTensor(question)
    return image, question, length, answer, index
