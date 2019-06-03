import torch
import torch.nn as nn
import torch.nn.functional as F
from .tcn_utils import TemporalConvNet
from .initial import initialize_weights
from collections import OrderedDict

class CITCNN(nn.Module):

  def __init__(self, text_dim, input_dim=3, global_pool=False):
    super(CITCNN, self).__init__()
    # Input 96 x 96
    self.conv1 = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(input_dim,   64, kernel_size=3, stride=1, padding=1)),
                ('bn1',   nn.GroupNorm(32, 64))]))
    self.gamma1 = nn.Linear(text_dim, 64)
    self.beta1  = nn.Linear(text_dim, 64)
    self.relu1 = nn.Sequential(OrderedDict([
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2))]))
    self.conv2 = nn.Sequential(OrderedDict([
                ('conv2', nn.Conv2d(64,   96, kernel_size=3, stride=1, padding=1)),
                ('bn2',   nn.GroupNorm(32, 96))]))
    self.gamma2 = nn.Linear(text_dim, 96)
    self.beta2  = nn.Linear(text_dim, 96)
    self.relu2 = nn.Sequential(OrderedDict([
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2))]))
    self.conv3 = nn.Sequential(OrderedDict([
                ('conv3', nn.Conv2d(96,  128, kernel_size=3, stride=1, padding=1)),
                ('bn3',   nn.GroupNorm(32, 128))]))
    self.gamma3 = nn.Linear(text_dim, 128)
    self.beta3  = nn.Linear(text_dim, 128)
    self.relu3 = nn.Sequential(OrderedDict([
                ('relu3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(kernel_size=2, stride=2))]))
    self.conv4 = nn.Sequential(OrderedDict([
                ('conv4', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                ('bn4',   nn.GroupNorm(32, 256))]))
    self.gamma4 = nn.Linear(text_dim, 256)
    self.beta4  = nn.Linear(text_dim, 256)
    self.relu4 = nn.ReLU(inplace=True)
    self.final_dim = 256
    self.global_pool = global_pool

  def forward(self, images, texts):
    conv1 = self.conv1(images)
    gamma1 = self.gamma1(texts).unsqueeze(2).unsqueeze(2)
    beta1 = self.beta1(texts).unsqueeze(2).unsqueeze(2)
    feat1 = conv1 * gamma1 + beta1
    relu1 = self.relu1(feat1)

    conv2 = self.conv2(relu1)
    gamma2 = self.gamma2(texts).unsqueeze(2).unsqueeze(2)
    beta2 = self.beta2(texts).unsqueeze(2).unsqueeze(2)
    feat2 = conv2 * gamma2 + beta2
    relu2 = self.relu2(feat2)

    conv3 = self.conv3(relu2)
    gamma3 = self.gamma3(texts).unsqueeze(2).unsqueeze(2)
    beta3 = self.beta3(texts).unsqueeze(2).unsqueeze(2)
    feat3 = conv3 * gamma3 + beta3
    relu3 = self.relu3(feat3)

    conv4 = self.conv4(relu3)
    gamma4 = self.gamma4(texts).unsqueeze(2).unsqueeze(2)
    beta4 = self.beta4(texts).unsqueeze(2).unsqueeze(2)
    feat4 = conv4 * gamma4 + beta4
    feature = self.relu4(feat4)

    if self.global_pool:
      feature = torch.mean(feature, dim=2)
      feature = torch.mean(feature, dim=2)
    return feature


class CNN_FAIT(nn.Module):

  def __init__(self, vocab_size, num_class):
    super(CNN_FAIT, self).__init__()
    self.word_dim = 512
    self.fin_dim  = 512
    self.embedding = nn.Embedding(vocab_size, self.word_dim)
    self.tcn = TemporalConvNet(self.word_dim, [self.fin_dim], 4, dropout=0.2, global_pool=True)
    self.cnn = CITCNN(self.fin_dim, 3, True)
    
    self.fc = nn.Sequential(
                        nn.Linear(self.cnn.final_dim, 512),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(512, 512),
                        nn.ReLU(True),
                        nn.Dropout())
    self.classifier = nn.Linear(512, num_class)
    self.apply(initialize_weights)

  def forward(self, images, sequences, lengths, lengths_var):
    word_embed = self.embedding(sequences)
    text_features = self.tcn(word_embed.transpose(1, 2))
  
    image_features = self.cnn(images, text_features)
  
    features = image_features
    #features = torch.cat((image_features, sentence_features), 1)
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
