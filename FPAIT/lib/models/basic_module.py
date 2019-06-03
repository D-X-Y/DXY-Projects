import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SimpleCNN(nn.Module):

  def __init__(self, input_dim=3, global_pool=False):
    super(SimpleCNN, self).__init__()
    # Input 96 x 96
    self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(input_dim,   64, kernel_size=3, stride=1, padding=1)),
                ('bn1',   nn.GroupNorm(32, 64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv2', nn.Conv2d(64,   96, kernel_size=3, stride=1, padding=1)),
                ('bn2',   nn.GroupNorm(32, 96)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv3', nn.Conv2d(96,  128, kernel_size=3, stride=1, padding=1)),
                ('bn3',   nn.GroupNorm(32, 128)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv4', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                ('bn4',   nn.GroupNorm(32, 256)),
                ('relu4', nn.ReLU(inplace=True))
        ]))
    self.final_dim = 256
    self.global_pool = global_pool

  def forward(self, x):
    feature = self.features(x)
    if self.global_pool:
      feature = torch.mean(feature, dim=2)
      feature = torch.mean(feature, dim=2)
    return feature
