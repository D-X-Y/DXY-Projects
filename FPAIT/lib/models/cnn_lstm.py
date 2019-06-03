import torch
import torch.nn as nn
import torch.nn.functional as F
from .initial import initialize_weights
from .basic_module import SimpleCNN
from collections import OrderedDict


class CNN_RNN(nn.Module):

  def __init__(self, vocab_size, num_class, rnn_type):
    super(CNN_RNN, self).__init__()
    self.word_dim = 512
    self.fin_dim  = 512
    self.embedding = nn.Embedding(vocab_size, self.word_dim)
    self.cnn = SimpleCNN(3, True)
    if rnn_type == 'LSTM':
      self.rnn = nn.LSTM(self.word_dim, self.fin_dim, num_layers=1, batch_first=True, dropout=0)
    elif rnn_type == 'GRU':
      self.rnn = nn.GRU(self.word_dim, self.fin_dim, num_layers=1, batch_first=True, dropout=0)
    elif rnn_type.lower() == 'vanilla':
      self.rnn = nn.RNN(self.word_dim, self.fin_dim, num_layers=1, batch_first=True, dropout=0)
    else:
      raise TypeError('Unknow rnn type: {:}'.format(rnn_type))

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
    packed = torch.nn.utils.rnn.pack_padded_sequence(word_embed, lengths, batch_first=True)
    rnn_output, _ = self.rnn(packed)
    rnn_padded = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    lengths_var = lengths_var.view(-1, 1, 1)
    lengths_var = lengths_var.expand(lengths_var.size(0), 1, self.word_dim)-1
    sentence_features = torch.gather(rnn_padded[0], 1, lengths_var).squeeze(1)
  
    image_features = self.cnn(images)
  
    features = torch.cat((image_features, sentence_features), 1)
    
    features = self.fc(features)
    predictions = self.classifier(features)
    
    return predictions

  def fix_parameters(self, base_lr, ratio=0):
    params_dict = [ {'params': self.cnn.parameters(),       'lr': base_lr*ratio},
                    {'params': self.embedding.parameters(), 'lr': base_lr*ratio},
                    {'params': self.rnn.parameters(),       'lr': base_lr*ratio},
                    {'params': self.fc.parameters(),        'lr': base_lr*ratio},
                    {'params': self.classifier.parameters(),'lr': base_lr},
                  ]
    return params_dict


class CNN_SIMPLE(nn.Module):

  def __init__(self, vocab_size, num_class):
    super(CNN_SIMPLE, self).__init__()
    self.word_dim = 512
    self.fin_dim = 512
    self.embedding = nn.Embedding(vocab_size, self.word_dim)
    self.cnn = SimpleCNN(3)
    self.rnn = nn.Sequential(
                  nn.Linear(self.word_dim, self.fin_dim)
               )

    self.fc         = nn.Sequential(
                        nn.Linear(self.fin_dim + self.cnn.final_dim, 512),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(512, 512),
                        nn.ReLU(True),
                        nn.Dropout())
    self.classifier = nn.Linear(512, num_class)
    self.apply(initialize_weights)

  def forward(self, images, sequences, lengths, lengths_var):
    word_embeds = self.embedding(sequences)
    mean_feas = []
    for word_embed, length in zip(word_embeds, lengths):
      mean_feature = torch.mean(word_embed[:length], 0)
      mean_feas.append(mean_feature)
    batch_means = torch.stack(mean_feas)
    sentence_features = self.rnn(batch_means)

    import pdb
    pdb.set_trace()
    image_features = self.cnn(images).squeeze(2).squeeze(2)
    features = torch.cat((image_features, sentence_features), 1)

    features = self.fc(features)
    predictions = self.classifier(features)
    
    return predictions

  def fix_parameters(self, base_lr, ratio=0):
    params_dict = [ {'params': self.cnn.parameters(),       'lr': base_lr*ratio},
                    {'params': self.embedding.parameters(), 'lr': base_lr*ratio},
                    {'params': self.rnn.parameters(),       'lr': base_lr*ratio},
                    {'params': self.fc.parameters(),        'lr': base_lr*ratio},
                    {'params': self.classifier.parameters(),'lr': base_lr},
                  ]
    return params_dict
