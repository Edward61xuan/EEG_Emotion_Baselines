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
# =========================================================
###########################################################
# ===================== DGCNN Model =======================

def compute_normalized_laplacian(W):
    D = torch.diag(torch.sum(W, dim=1) + 1e-6)
    L = D - W
    D_inv_sqrt = torch.diag(torch.pow(torch.sum(W, dim=1), -0.5))
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    eigval_max = 2.0
    L_tilde = (2.0 * L_norm) / eigval_max - torch.eye(W.size(0), device=W.device)
    return L_tilde

def chebyshev_polynomials(L_tilde, K):
    N = L_tilde.size(0)
    T_k = []
    T_k.append(torch.eye(N, device=L_tilde.device))
    if K > 1:
        T_k.append(L_tilde)
    for k in range(2, K):
        T_k.append(2 * torch.matmul(L_tilde, T_k[-1]) - T_k[-2])
    return T_k

class ChebConv(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(ChebConv, self).__init__()
        self.K = K
        self.linear = nn.Parameter(torch.Tensor(K, in_features, out_features))
        nn.init.xavier_uniform_(self.linear)

    def forward(self, x, T_k_list):
        out = 0
        for k in range(self.K):
            T_k = T_k_list[k]
            Tx = torch.einsum('nm,bmf->bnf', T_k, x)
            out += torch.matmul(Tx, self.linear[k])
        return out

class DGCNNBlock(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, K, num_nodes):
        super(DGCNNBlock, self).__init__()
        self.K = K
        self.num_nodes = num_nodes
        self.W_star = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_normal_(self.W_star)
        self.cheb_conv = ChebConv(in_features, hidden_features, K)
        self.conv1x1 = nn.Linear(hidden_features, hidden_features)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        # x: [B, N, Fin]
        W_pos = F.relu(self.W_star)
        L_tilde = compute_normalized_laplacian(W_pos)
        T_k_list = chebyshev_polynomials(L_tilde, self.K)
        out = self.cheb_conv(x, T_k_list)   # [B, N, Fout]
        out = self.conv1x1(out)
        out = self.relu(out)
        out = out.permute(0, 2, 1)          # [B, Fout, N]
        out = self.global_pool(out).squeeze(-1)   # [B, Fout]
        logits = self.fc(out)
        return logits
