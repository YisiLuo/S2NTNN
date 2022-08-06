import torch
import scipy.io
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor

def make(x,x_n3): # This function is to expand the observed data with small n_3 (e.g., MSI) to bigger n_3
    y = torch.ones(x.shape[0],x.shape[1],x_n3).type(dtype)
    for i in range(int(x_n3/x.shape[2])):
        i *= 3
        y[:,:,i:i+x.shape[2]] = x
    return y

def psnr3d(x,y): 
    ps = 0
    for i in range(x.shape[2]):
        ps += peak_signal_noise_ratio(x[:,:,i], y[:,:,i])
    return ps/x.shape[2]

class permute_in(nn.Module):
    def __init__(self, dim):
        super(permute_in, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.permute(2,0,1)
        x_in = x.view(1,self.dim,x.shape[1],x.shape[2])
        return x_in
    
class permute_out(nn.Module):
    def __init__(self, dim):
        super(permute_out, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.view(self.dim,x.shape[2],x.shape[3])
        x_out = x.permute(1,2,0)
        return x_out