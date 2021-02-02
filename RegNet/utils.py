import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import re


def get_gc(dna):
    # assume order is ACGT
    percent = dna.sum(1) / dna.shape[1]
    return(float(percent[1] + percent[2]))


def get_cg(dna):
    string = reverse_one_hot(dna)
    return(len(re.findall("CG", string)))


def one_hot(dna):
    dna = dna.upper()
    d = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
    oh = torch.nn.functional.one_hot(torch.tensor([d[x] for x in dna]))
    return(oh[:,0:4].t().type(torch.FloatTensor))


def reverse_one_hot(oh):
    d = ['A','C','G','T','N']
    # account for N
    ints = np.where(oh.numpy().max(0) != 0, np.argmax(oh, 0), 4).tolist()
    string = [d[x] for x in ints]
    return(''.join(string))


def get_final_layer_input_size(in_width, pool_sizes, n_kernels):
    """Return size of final layer after series of same convolutions with max pooling between"""
    out_size = in_width
    for i in range(len(pool_sizes)):
        out_size = int(out_size / pool_sizes[i])
    out_size = out_size * n_kernels[i]
    return out_size


# Pearson loss function from AI-TAC: https://github.com/smaslova/AI-TAC/blob/master/code/aitac.py
def pearson_loss(x,y):
    mx = torch.mean(x, dim=1, keepdim=True)
    my = torch.mean(y, dim=1, keepdim=True)
    xm, ym = x - mx, y - my
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = torch.sum(1-cos(xm,ym))
    return loss 

# def pearson_loss(x,y):
#     y = y.type(torch.FloatTensor)
#     x = x.type(torch.FloatTensor)
#     vx = x - torch.mean(x)
#     vy = y - torch.mean(y)
#     cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
#     return(cost)

class Exponential(nn.Module):
    """Exponential activation function for use with nn.Sequential"""
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return torch.exp(input)