import torch.nn as nn
from utils import orthogonal_init
from functools import reduce 
from utils import conv_shape, conv_seq_shape

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim, w, h):




    super().__init__()

    conv_1 = dict(kernel_size=8, stride=4)
    conv_2 = dict(kernel_size=4, stride=2)
    conv_3 = dict(kernel_size=3, stride=1)
    seq = [conv_1, conv_2, conv_3]
    in_linear = conv_seq_shape(w, seq) * conv_seq_shape(h, seq)
    conv_out_channels = 64
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, **conv_1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, **conv_2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=conv_out_channels, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=in_linear * conv_out_channels, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)