import os, pdb, json, sys, torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

lib_path = str((Path(__file__).parent.resolve() / 'lib').resolve())
if lib_path not in sys.path: sys.path.insert(0, lib_path)

from logger import Logger
from datasets import MetaQATask, VQAMetaDataset
from models import obtain_fait_model


class Option(object):
  def __init__(self, arch, vocab_size, num_classes, use_cuda):
    self.arch = arch
    self.vocab_size = vocab_size
    self.num_classes = num_classes
    self.use_cuda = use_cuda


def demo_codes(dataset, pth_data_path, n_way=10, k_shot=5):
  log = Logger('outputs/{:}-N{:}-K{:}'.format(dataset, n_way, k_shot))
  meta_task = MetaQATask(dataset, pth_data_path, n_way, k_shot, log)
  vocab_size = len(meta_task.all_words)

  transform  = transforms.Compose([transforms.RandomHorizontalFlip(), \
                      transforms.RandomResizedCrop(96), \
                      transforms.ToTensor(), \
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
  meta_train_dataset = VQAMetaDataset(meta_task, transform, 15, vocab_size, True) # training
  meta_valid_dataset = VQAMetaDataset(meta_task, transform, 15, vocab_size, True) # testing
  meta_train_loader = torch.utils.data.DataLoader(meta_train_dataset, batch_size=32, shuffle=True , num_workers=8, pin_memory=True)
  meta_valid_loader = torch.utils.data.DataLoader(meta_valid_dataset, batch_size=32, shuffle=True , num_workers=8, pin_memory=True)

  model_config = Option('cnn_fait', vocab_size, n_way, True)
  model = obtain_fait_model(model_config)
  optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)


  num_epochs = 50
  for iepoch in range(num_epochs):
    print ('start {:03d}-th epoch'.format(iepoch))
    for idx, (training_images, training_questions, training_lengths, training_answers, training_indexes, \
         testing_images, testing_questions, testing_lengths, testing_answers, testing_indexes) in enumerate(meta_train_loader):
      # this meta_train_loader returns a batch of tasks, which task has n*k training samples and test samlpes
      batch_size = training_images.size(0)
      for itask, (cur_train_images, cur_train_ques, cur_train_ans) in enumerate(zip(training_images, training_questions, training_answers)):
        cur_train_images, cur_train_ques, cur_train_ans = cur_train_images.cuda(), cur_train_ques.cuda(), cur_train_ans.cuda()
        cur_outs = model(cur_train_images, cur_train_ques, None, None)
        cur_train_loss = torch.nn.functional.cross_entropy(cur_outs,  cur_train_ans, reduction='mean')
        # this is just an examlpe written about three years after the original project.
        # you can implement your optimization algorithms here for `meta learning for VQA and IC`.
        import pdb; pdb.set_trace()


if __name__ == '__main__':
  demo_codes('toronto-coco-qa', './data/Toronto-COCO-QA/object.pth') # VQA

  demo_codes('ms-coco-ic', './data/COCO-Caption/few-shot-coco.pth')  # Image-Caption
