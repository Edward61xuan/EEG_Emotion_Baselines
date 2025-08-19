import argparse
import os
gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io
import wandb

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
# from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
# from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# ========================================================
##########################################################
# ===================== EEGNet Model =====================

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=None,
                 norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if F2 is None:
            F2 = F1 * D

        if dropoutType == 'Dropout':
            self.dropout = nn.Dropout(p=dropoutRate)
        elif dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d(p=dropoutRate)
        else:
            raise ValueError("dropoutType must be one of 'Dropout' or 'SpatialDropout2D'")

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise Convolution
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1),
                                       groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        # Block 2: Separable Convolution
        self.separableConv_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                             padding=(0, 8), groups=F1 * D,
                                             bias=False)
        self.separableConv_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        # Final classification layer
        # self.classifier = nn.Linear(F2 * (Samples // 32), nb_classes)  
        self.classifier = nn.LazyLinear(nb_classes)  # 使用 LazyLinear 自动推断输入维度
        # 因为经过 pool(1,4) 再 pool(1,8) → 时间维度缩小 32 倍

        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.avgpool2 = nn.AvgPool2d((1, 8))

    def forward(self, x):
        
        x = x.unsqueeze(1)  # to [B, 1, Chans, Samples]
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.separableConv_depth(x)
        x = self.separableConv_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # Classifier
        out = self.classifier(x)
        return F.log_softmax(out, dim=1)