"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from config import cfg
import torch.nn as nn
from math import sqrt
import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat
from torch.nn.modules import Module
from  torch.utils.checkpoint  import checkpoint


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL,'BNFUNC')
    normalizationLayer = layer(in_channels)
    return normalizationLayer


def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, groups=1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class GroupNorm(nn.Module):
    def __init__(self, in_channels, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,in_channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,in_channels,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        # assert C % G == 0
        #
        x = x.view(N,G,-1)
        x = (x-x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True)+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias