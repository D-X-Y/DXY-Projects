import os, pdb, json, sys, torch
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
api_path = str((Path(__file__).parent.resolve() / '..' / 'cocoapi').resolve())
if api_path not in sys.path: sys.path.insert(0, api_path)
print ('cocoapi : {:}'.format(api_path))
lib_path = str((Path(__file__).parent.resolve() / '..' / 'lib').resolve())
if lib_path not in sys.path: sys.path.insert(0, lib_path)

from datasets import TICSample
SEED = 112
def set_random():
  random.seed(SEED)
  np.random.seed(SEED)

def check_double_words(sentence, blanks):
  total, xx = 0,0 
  for blank in blanks:
    if blank in sentence: total += 1
  for word in sentence:
    if word in blanks: xx += 1
  return total, xx

def main(coco_dir, cur_dir):
  cap_coco_path = cur_dir / 'cap_coco.json'
  with open(cap_coco_path, 'r') as cfile:
    cap_coco_data = json.load(cfile)
  cfile.close()
  length = len(cap_coco_data)
  all_words2idx = dict()
  for cap_data in cap_coco_data:
    for sentence in cap_data:
      for word in sentence:
        if word not in all_words2idx:
          all_words2idx[word] = len(all_words2idx)
  print ('{:} images with {:} words'.format(length, len(all_words2idx)))

  dic_coco_path = cur_dir / 'dic_coco.json'
  with open(dic_coco_path, 'r') as cfile:
    dic_coco_data = json.load(cfile)
  cfile.close()
  images = dic_coco_data['images']
  assert len(images) == length, '{:} vs {:}'.format(len(images), length)
  wtod   = dic_coco_data['wtod']
  wtol   = dic_coco_data['wtol']
  ix_to_word = dic_coco_data['ix_to_word']

  coco_class_all, concepts = [], []
  coco_class_name = open(cur_dir / 'coco_class_name.txt', 'r')
  for line in coco_class_name:
      coco_class = line.rstrip("\n").split(', ')
      coco_class_all.append(coco_class)
      concepts += coco_class
  coco_class_name.close()
  concepts = [x for x in concepts if ' ' not in x]

  print ('COCO-DIR : {:}'.format(coco_dir))
  assert os.path.exists(coco_dir), 'COCO DIR does not exist : {:}'.format(coco_dir)
  all_captions = []
  for sentences, image in zip(cap_coco_data, images):
    file_path = coco_dir / 'trainval2014' / image['file_path'].split('/')[1]
    assert file_path.exists(), '{:}'.format(file_path)
    for sentence in sentences:
      tic = TICSample(file_path, sentence)
      if len(tic) < 15:
        all_captions.append(tic)

  print ('original captions : {:}'.format(len(all_captions)))
  print ('total blank-concepts : {:}'.format(len(concepts)))
  # filter
  filtered_captions = []
  for caption in all_captions:
    num, nmm = check_double_words(caption.sentence, concepts)
    if num == 1 and nmm == 1:
      filtered_captions.append( caption )

  for caption in filtered_captions:
    caption.replace(concepts)
  print ('filtered captions : {:}'.format(len(filtered_captions)))
  all_lengths = np.array([len(x) for x in filtered_captions])
  print ('LENGTH : min={:}, max={:}, mean={:}, std={:}'.format(all_lengths.min(), all_lengths.max(), all_lengths.mean(), all_lengths.std()))
    
  all_words2idx = dict()
  for caption in filtered_captions:
    for word in caption.sentence:
      if word not in all_words2idx:
        all_words2idx[word] = len(all_words2idx)
  print ('<-> FINAL : {:} captions with {:} words'.format(len(filtered_captions), len(all_words2idx)))

  for concept in concepts:
    assert concept not in all_words2idx, 'concept : {:}'.format(concept)
  # collect
  set_random()
  concept2idx = defaultdict(list)
  for idx, caption in enumerate(filtered_captions):
    assert caption.blank is not None, 'caption [{:03d}] = {:}'.format(idx, caption)
    concept2idx[ caption.blank ].append(caption)
  print ('Before filter : {:}'.format( len(concept2idx) ))
  temp = [(key,value) for key, value in concept2idx.items() if len(value) >=10]
  concept2idx = dict(temp)
  print ('After  filter : {:}'.format( len(concept2idx) ))

  concepts = [ key for key, value in concept2idx.items()]
  shuffle_concepts = deepcopy(concepts)
  random.shuffle(shuffle_concepts)
  print ('First-Two : {:} {:}'.format(shuffle_concepts[0], shuffle_concepts[1]))
  train_keys = []
  current, total = 0, len(filtered_captions)
  for key in shuffle_concepts:
    current += len(concept2idx[key])
    train_keys.append(key)
    if current > int(total * 0.8): break
  train_keys = set(train_keys)
  test_keys = set(shuffle_concepts).difference(train_keys)

  training, testing = [], []
  for key in train_keys:
    training += concept2idx[key]
  for key in test_keys:
    testing  += concept2idx[key]
  print ('Training : {:} keys for {:} captions'.format( len(train_keys), len(training)))
  print ('Testing  : {:} keys for {:} captions'.format( len( test_keys), len( testing)))
  
  save_path = cur_dir / 'few-shot-coco.pth'
  all_words, all_blanks = set(), set()
  words2index = {}
  for caption in training + testing:
    for word in caption.sentence:
      all_words.add(word)
    all_blanks.add(caption.blank)
  for index, word in enumerate(all_words):
    words2index[word] = index
  print ('Finally words : {:}, blanks : {:}'.format( len(words2index), len(all_blanks) ))
  icdata = {'all_words':   all_words,
            'all_blanks':  all_blanks,
            'words2index': words2index,
            'training' :   training,
            'testing'  :   testing}
  torch.save(icdata, save_path)
  print ('Save into {:}'.format(save_path))
  blanks = set()
  for caption in training:
    blanks.add(caption.blank)
  print ('Training blanks : {:}'.format(len(blanks)))
  blanks = set()
  for caption in testing:
    blanks.add(caption.blank)
  print ('Testing  blanks : {:}'.format(len(blanks)))
  print ('Total has {:} image-caption pairs'.format(len(training+testing)))

if __name__ == '__main__':
  root_dir = os.environ['HOME']
  data_dir = Path(root_dir) / 'datasets' / 'MS-COCO'
  current_dir = Path(__file__).parent.resolve() / 'COCO-Caption'
  main(data_dir, current_dir)
