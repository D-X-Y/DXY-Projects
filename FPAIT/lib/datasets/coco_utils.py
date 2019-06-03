import re, pdb
from pathlib import Path
from copy import deepcopy

def load_caption(root, subset):
  root = Path(root)
  coco_ann_file = root / 'annotations' / 'instances_{}.json'.format(subset)
  coco = COCO(coco_ann_file)
  coco_caps_file = root / 'annotations' / 'captions_{}.json'.format(subset)
  coco_caps = COCO(coco_caps_file)

  # display COCO categories and supercategories
  cats = coco.loadCats(coco.getCatIds())
  names = [cat['name'] for cat in cats]
  print ('COCO categories [{:}] : \n{:}\n'.format(len(names), ' '.join(names)))

  supernames = set([cat['supercategory'] for cat in cats])
  print('COCO supercategories [{:}]: \n{:}'.format(len(supernames), ' '.join(supernames)))
  return coco, coco_cap


chars = 'qwertyuiopasdfghjklzxcvbnm ,\'-.!?`:;#*@&$/\\1234567890'
sepss = '.,!?;: '


class TQASample(object):

  def __init__(self, img_id, question, answer, qatype):
    self.img_id = int(img_id)
    self.image_info = None
    self.question = self.clean_str(question)
    assert all(x in chars for x in self.question), '{:}'.format(question)
    self.answer = self.clean_str(answer)
    self.qatype = int(qatype)
  
  def clean_str(self, string):
    string = string.strip().strip('\n').lower()
    return string

  def set_info(self, info):
    if isinstance(info, list):
      assert len(info) == 1, 'info : {:}'.format(len(info))
      info = info[0]
    self.image_info = deepcopy(info)

  def split(self, string=None):
    if string is None:
      string = self.question
    string = self.clean_str(string)
    for char in ',.;:':
      string = ' {:} '.format(char).join( string.split(char) )
    strsss = string.split(' ')
    strsss = [x for x in strsss if x != '']
    return strsss

  def __len__(self):
    return len(self.split())

  def get_name(self):
    return self.image_info['file_name']
  
  def __repr__(self):
    return ('{name}(question=\'{question}\', answer=\'{answer}\', type={qatype}, image={image_info})'.format(name=self.__class__.__name__, **self.__dict__))

class TICSample(object):

  def __init__(self, file_path, sentence):
    self.image = deepcopy(file_path)
    self.back_sentence = deepcopy(sentence)
    self.sentence = deepcopy(sentence)
    assert isinstance(sentence, list), 'TIC Type : {:}'.format(sentence)
    self.blank = None

  def reset(self):
    self.sentence = deepcopy(self.back_sentence)
    self.blank = None
  
  def replace(self, concepts, blank_holder='<BLANK>'):
    place = -1
    for idx, word in enumerate(self.sentence):
      if word in concepts:
        if place == -1:
          place = idx
        else:
          raise ValueError('{:} vs {:} : {:}'.format(place, idx, self.sentence))
    self.blank = self.sentence[place]
    self.sentence[place] = blank_holder
      
  def __len__(self):
    return len(self.sentence)

  def get_name(self):
    return str(self.image)
  
  def __repr__(self):
    return ('{name}(image=\'{image}\', sentence=\'{sentence}\', blank=\'{blank}\')'.format(name=self.__class__.__name__, **self.__dict__))



class SimpleQA(object):

  def __init__(self, words_index, data, from_vqa=True):
    if from_vqa:
      data = deepcopy(data)
      self.ori_question = data.question
      self.question = [words_index[x] for x in data.split()]
      self.answer = data.answer
      self.image = data.image_info['image']
    else:         # from fill-in-the-blank
      data = deepcopy(data)
      self.ori_question = None
      self.question = [words_index[x] for x in data.sentence]
      self.answer = data.blank
      self.image = data.image

  def check(self, wmax):
    assert len(self.question) > 0, '{:}'.format(self)
    for x in self.question:
      assert x>=0 and x<wmax, '{:}'.format(self)

  def __len__(self):
    return len(self.question)

  def __repr__(self):
    return ('{name}(question=\'{question}\', answer=\'{answer}\', image={image}, base={ori_question})'.format(name=self.__class__.__name__, **self.__dict__))
