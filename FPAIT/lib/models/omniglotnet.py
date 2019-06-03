import numpy as np
import random
import math
import torch
import torch.nn as nn
from collections import OrderedDict

class OmniglotNet(nn.Module):
  '''
  The base model for few-shot learning on Omniglot
  '''

  def __init__(self, num_classes):
    super(OmniglotNet, self).__init__()
    # input 28x28
    self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, 3)),
                ('bn1',   nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2,2)),
                ('conv2', nn.Conv2d(64,64,3)),
                ('bn2',   nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2,2)),
                ('conv3', nn.Conv2d(64,64,3)),
                ('bn3',   nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(2,2))
        ]))
    self.fc = nn.Linear(64, num_classes)
    
    # Initialize weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        #m.bias.data.zero_() + 1
        m.bias.data = torch.ones(m.bias.data.size())

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 64)
    x = self.fc(x)
    return x

def omniglotnet(num_classes):
  net = OmniglotNet(num_classes)
  return net
