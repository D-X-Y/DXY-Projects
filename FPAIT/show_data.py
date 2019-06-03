import os, pdb, json, sys, torch
import numpy as np
from pathlib import Path
lib_path = str((Path(__file__).parent.resolve() / 'lib').resolve())
if lib_path not in sys.path: sys.path.insert(0, lib_path)
from datasets import TICSample, TQASample


def show_vqa(data):
  all_words = data['all_words']
  words_index = data['words_index']
  all_answers = data['all_answers']
  answers_index = data['answers_index']
  training = data['training']
  testing  = data['testing']
  print ('Few-shot VQA:')
  print ('  ->> {:} training samples, {:} testing samples'.format(len(training), len(testing)))
  for idx, x in enumerate(training):
    if idx < 3:
      print ('  ->> {:}/{:} : {:}'.format(idx, len(training), x))

def show_ic(data):
  all_words = data['all_words']
  all_blanks = data['all_blanks']
  words2index = data['words2index']
  training = data['training']
  testing  = data['testing']
  print ('Few-shot Image Caption:')
  print ('  ->> {:} training samples, {:} testing samples'.format(len(training), len(testing)))
  for idx, x in enumerate(training):
    if idx < 3:
      print ('  ->> {:}/{:} : {:}'.format(idx, len(training), x))
    

if __name__ == '__main__':
  vqa_list_path = './data/Toronto-COCO-QA/object.pth'
  vqa_list = torch.load(vqa_list_path)
  show_vqa(vqa_list)

  print ('')
  ic_list_path = './data/COCO-Caption/few-shot-coco.pth'
  ic_list = torch.load(ic_list_path)
  show_ic(ic_list)
