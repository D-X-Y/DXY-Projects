import os, sys, pdb, torch
import random, numpy as np
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
api_path = str((Path(__file__).parent.resolve() / '..' / 'cocoapi').resolve())
if api_path not in sys.path: sys.path.insert(0, api_path)
print ('cocoapi : {:}'.format(api_path))
from pycocotools.coco import COCO
lib_path = str((Path(__file__).parent.resolve() / '..' / 'lib').resolve())
if lib_path not in sys.path: sys.path.insert(0, lib_path)
from datasets import TQASample
SEED = 112
types_of_questions=['object', 'number', 'color', 'location']

def set_random():
  random.seed(SEED)
  np.random.seed(SEED)

def read_lines(path):
  assert path.exists(), '{:} does not exist'.format(path)
  with open(path, 'r') as cfile:
    content = [line.strip() for line in cfile.readlines()]
  return content

def load_data(data_dir):
  img_ids   = read_lines( data_dir / 'img_ids.txt' )
  answers   = read_lines( data_dir / 'answers.txt' )
  questions = read_lines( data_dir / 'questions.txt' )
  qatypes   = read_lines( data_dir / 'types.txt' )

  datas = []
  for img_id, answer, question, qatype in zip(img_ids, answers, questions, qatypes):
    sample = TQASample(img_id, question, answer, qatype)
    datas.append( sample )
  return datas

def ayalysis_qatype(datas, qatype, threshold, save_path):
  selected = []
  for data in datas:
    if data.qatype == qatype:
      if len(data) < 15:
        selected.append( deepcopy(data) )
  print ('[{:}] :: select {:} from {:} datas by qatype={:}'.format(save_path, len(selected), len(datas), qatype))

  all_answers = defaultdict(lambda: 0)
  all_words   = defaultdict(lambda: 0)
  for sample in selected:
    all_answers[sample.answer] += 1
    for x in sample.split():
      all_words[x] += 1
  word_freq = np.array([all_words[x] for x in all_words])

  print ('  {:} different answers for {:}, total words : {:}'.format(len(all_answers), save_path, len(word_freq)))
  final_selected = []
  for sample in selected:
    answer = sample.answer
    if all_answers[sample.answer] < threshold: continue
    bool_words = True
    for word in sample.split():
      if all_words[word] < 3: bool_words = False
    if bool_words:
      final_selected.append(sample)
  print ('  By threshold={:}, left {:} from {:}'.format(threshold, len(final_selected), len(selected)))

  all_answers = defaultdict(lambda: 0)
  all_words   = defaultdict(lambda: 0)
  for sample in final_selected:
    all_answers[sample.answer] += 1
    for x in sample.split():
      all_words[x] += 1
  
  total = len(final_selected)
  print ('  {:} different answers, and {:} words for {:}'.format(len(all_answers), len(all_words), save_path))
  all_lengths = np.array([len(x) for x in final_selected])

  train_keys = []
  current = 0
  keyss = [key for key in all_answers]
  set_random()
  random.shuffle(keyss)
  for key in keyss:
    current += all_answers[key]
    train_keys.append(key)
    if current > int(total * 0.8):
      break

  training, testing = [], []
  for sample in final_selected:
    sample.image_info['image'] = Path(data_dir) / 'trainval2014' / sample.image_info['file_name']
    assert sample.image_info['image'].exists(), '{:}'.format(sample.image_info['image'])
    if sample.answer in train_keys:
      training.append(sample)
    else:
      testing.append(sample)

  print ('  Training : {:} -> {:}, Testing : {:} -> {:}'.format(len(train_keys), len(training), len(all_answers)-len(train_keys), len(testing)))
  words_index = {}
  for index, key in enumerate(all_words):
    words_index[key] = index
  answers_index = {}
  for index, key in enumerate(all_answers):
    answers_index[key] = index
    

  qadata = {'all_words'  :   dict(all_words),
            'words_index':   words_index,
            'all_answers':   dict(all_answers),
            'answers_index': answers_index,
            'training'   :   training,
            'testing'    :   testing}

  torch.save(qadata, save_path)
  
  # samples
  testkeys = set()
  for sample in testing:
    testkeys.add( sample.answer )
  print( testkeys )


def main(coco_dir):
  print ('MS-COCO : {:}'.format(coco_dir))
  coco = COCO(coco_dir/'annotations'/'instances_train2014.json')
  train_dir = Path('./Toronto-COCO-QA/train')
  train_datas = load_data(train_dir)

  for sample in train_datas:
    sample.set_info( coco.loadImgs(sample.img_id) )

  coco = COCO(coco_dir/'annotations'/'instances_val2014.json')
  test_dir  = Path('./Toronto-COCO-QA/test')
  test_datas = load_data(test_dir)
  for sample in test_datas:
    sample.set_info( coco.loadImgs(sample.img_id) )
  datas = train_datas + test_datas
  print ('there are {:} + {:} = {:} samlpes in total'.format( len(train_datas), len(test_datas), len(datas) ))

  for idx, type_name in enumerate(types_of_questions):
    ayalysis_qatype(datas, idx, 30, Path('./Toronto-COCO-QA/{:}.pth'.format(type_name)))
   

if __name__ == '__main__':
  root_dir = os.environ['HOME']
  data_dir = Path(root_dir) / 'datasets' / 'MS-COCO'
  main(data_dir)
