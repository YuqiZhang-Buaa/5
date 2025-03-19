import numpy as np
import torch
import torch.nn as nn
import math
from math import ceil
from mamba_latest.mamba_ssm import Mamba2_change
from mamba_latest.mamba_ssm import Mamba2
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn, einsum
import sys

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class SignificanceScorer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim//2, 1),  
            nn.ReLU(),
            nn.Conv1d(feature_dim//2, 1, 1)             
        )
    
    def forward(self, x):

        x = x.permute(0, 2, 1)        
        scores = self.scorer(x)        
        return scores.squeeze(1)       

def compute_morton(pos):

    L = pos.shape[0]
    device = pos.device
    morton_codes = []
    for i in range(L):
        x = pos[i, 0].item()
        y = pos[i, 1].item()
        max_bits = max(x.bit_length(), y.bit_length())
        code = 0
        for j in range(max_bits):
            code |= ((y >> j) & 1) << (2 * j)
            code |= ((x >> j) & 1) << (2 * j + 1)
        morton_codes.append(code)
    return torch.tensor(morton_codes, dtype=torch.int32, device=device)

def z_order_reorder(x, pos):
    morton_codes = compute_morton(pos)
    sorted_indices = torch.argsort(morton_codes)
    x_sorted = x[:, sorted_indices, :]
    return x_sorted

class PSMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2):
        super(SpeMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        self.sf = SignificanceScorer(512)
        self.survival = survival


        for _ in range(layer):
            self.layers_1.append(
                nn.Sequential(
                    # nn.LayerNorm(512),
                    Mamba2(
                        d_model=512,
                        d_state=64,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )
        for _ in range(layer):
            self.layers_2.append(
                nn.Sequential(
                    # nn.LayerNorm(512),
                    Mamba2(
                        d_model=512,
                        d_state=64,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  

        h = self._fc1(h[:,:,:-2])  
        list_p = x[:,:,-2:].squeeze(0)
        
        h_o = h


        h = z_order_reorder(h_o, list_p.to(torch.int32))
        for layer in self.layers_1:
            h_ = h
            h = layer[0](h)
            # h = layer[1](h)
            h = h + h_
        h_1 = h

        scores = self.sf(h_o)
        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        h = torch.gather(
            h_o, 
            1, 
            sorted_indices.unsqueeze(-1).expand(-1, -1, h_o.size(-1))
        )
        for layer in self.layers_2:
            h_ = h
            h = layer[0](h)
            # h = layer[1](h)
            h = h + h_
        h_2 = h


        h = torch.cat((h_1, h_2), dim=1) 

        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)


        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        # self.layers  = self.layers.to(device)
        self.layers_1  = self.layers_1.to(device)
        self.layers_2  = self.layers_2.to(device)
        self.sf = self.sf.to(device)
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
