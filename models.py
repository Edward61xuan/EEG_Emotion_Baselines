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
# ===========================================================
#############################################################
# ===================== Conformer Model =====================
class  PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, 8, (1, 125), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (22, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 8)),
            nn.Dropout2d(0.5)
        )

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (62, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # to [B, 1,Chans, Samples]
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(190, 1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.clshead_fc = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

# ! Rethink the use of Transformer for EEG signal
class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

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

# =========================================================
###########################################################
# ===================== CBraMod Model =====================

