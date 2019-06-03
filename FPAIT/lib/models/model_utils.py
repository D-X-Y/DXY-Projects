from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, numbers, numpy as np

def remove_module_dict(state_dict):
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
  return new_state_dict

def load_weight_from_dict(model, weight_state_dict, remove_prefix=False):
  if remove_prefix: weight_state_dict = remove_module_dict(weight_state_dict)
  all_parameter = model.state_dict()
  all_weights   = []
  finetuned_layer, random_initial_layer, broken_layer = [], [], []
  for key, value in all_parameter.items():
    if key in weight_state_dict:
      if value.size() == weight_state_dict[key].size():
        all_weights.append((key, weight_state_dict[key]))
        finetuned_layer.append(key)
      else:
        all_weights.append((key, value))
        broken_layer.append(key)
    else:
      all_weights.append((key, value))
      random_initial_layer.append(key)
  print ('==>[load_model] finetuned layers : {}'.format(finetuned_layer))
  print ('==>[load_model] keeped layers : {}'.format(random_initial_layer))
  if broken_layer:
    print ('==>[load_model] broken layers : {:}'.format(broken_layer))
  all_weights = OrderedDict(all_weights)
  model.load_state_dict(all_weights)
