import torch
import torch.nn as nn
import torch.nn.functional as F
from .tcn_utils import TemporalConvNet
from .initial import initialize_weights
from .basic_module import SimpleCNN
from collections import OrderedDict


class CNN_TCN(nn.Module):

  def __init__(self, vocab_size, num_class):
    super(CNN_TCN, self).__init__()
    self.word_dim = 512
    self.fin_dim  = 512
    self.embedding = nn.Embedding(vocab_size, self.word_dim)
    self.cnn = SimpleCNN(3, True)
    self.tcn = TemporalConvNet(self.word_dim, [self.fin_dim], 4, dropout=0.2, global_pool=True)
    
    self.fc = nn.Sequential(
                        nn.Linear(self.fin_dim + self.cnn.final_dim, 512),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(512, 512),
                        nn.ReLU(True),
                        nn.Dropout())
    self.classifier = nn.Linear(512, num_class)
    self.apply(initialize_weights)

  def forward(self, images, sequences, lengths, lengths_var):
    word_embed = self.embedding(sequences)
    sentence_features = self.tcn(word_embed.transpose(1, 2))
  
    image_features = self.cnn(images)
  
    features = torch.cat((image_features, sentence_features), 1)
    features = self.fc(features)
    predictions = self.classifier(features)
    
    return predictions

  def fix_parameters(self, base_lr, ratio=0):
    params_dict = [ {'params': self.cnn.parameters(),       'lr': base_lr*ratio},
                    {'params': self.embedding.parameters(), 'lr': base_lr*ratio},
                    {'params': self.tcn.parameters(),       'lr': base_lr*ratio},
                    {'params': self.fc.parameters(),        'lr': base_lr*ratio},
                    {'params': self.classifier.parameters(),'lr': base_lr},
                  ]
    return params_dict
