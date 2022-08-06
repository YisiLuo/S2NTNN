# import torchpwl (package of piecewise linear unit)
import torch
from torch import nn, optim 
from torch.autograd import Variable 
from torch.utils.data import DataLoader 
import os 
from utils import * 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt 
import scipy.io

dtype = torch.cuda.FloatTensor

class NoFC_3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(NoFC_3, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_channel,out_channel, bias = False),
                                   nn.LeakyReLU())
        
    def forward(self,x):
        return self.layer(x)

class SSNT(nn.Module): 
    def __init__(self,n_4,n_3):
        super(SSNT, self).__init__()
        self.f = nn.Sequential(  NoFC_3(n_3,n_4),
                                NoFC_3(n_4,n_4))
                                    
        self.g = nn.Sequential(NoFC_3(n_4,n_4),
                               NoFC_3(n_4,n_3))
                                    
    def forward(self, x):
        f_x = self.f(x)
        g_x = self.g(f_x)
        return f_x, g_x


data = 'data/pavia' # Dataset
c = '1' # Sampling rate
max_iter = 7000
F_norm = nn.MSELoss()

file_name = data+'p'+c+'.mat'
mat = scipy.io.loadmat(file_name)
X_np = mat["Nhsi"]
X = torch.from_numpy(X_np).type(dtype).cuda()

n_4 = 2 * X_np.shape[2] 

model = SSNT(n_4, X_np.shape[2]).cuda() 
s = sum([np.prod(list(p.size())) for p in model.parameters()]); 
print ('Number of params: %d' % s)

mask = torch.ones(X.shape).type(dtype)
mask[X == 0] = 0 
X[mask == 0] = 0

file_name = data+'p'+c+'linear_interpolation.mat'
mat = scipy.io.loadmat(file_name)
X_in_np = mat["input"]
X_in = torch.from_numpy(X_in_np).type(dtype).cuda()

params = []
params += [x for x in model.parameters()]

# Comment out the following two rows of code if you are dealing with video/MRI data.
X_in.requires_grad = True
params += [X_in]

optimizier = optim.Adam(params, lr=0.001, weight_decay=0) 
show = [8,8,8] # band
for iter in range(max_iter):
    
    lambda_ = 0.2 * 10e-7
    
    X_LR, X_Out = model(X_in)

    nuc = 0
    try:    
        nuc += lambda_*torch.norm(X_LR[:,:,int(iter%n_4)].cuda(),'nuc')
    except:
        nuc += lambda_*torch.norm(X_LR[:,:,int((iter+1)%n_4)].cuda(),'nuc')
    loss = nuc
    loss += F_norm(X_Out * mask,X * mask)
    
    optimizier.zero_grad()
    loss.backward(retain_graph=True)
    optimizier.step()

    if iter % 100 == 0:
        print('iteration:',iter)
        plt.subplot(131)
        plt.imshow(np.clip(np.stack((X[:,:,show[0]].cpu().detach().numpy(),
                             X[:,:,show[1]].cpu().detach().numpy(),
                             X[:,:,show[2]].cpu().detach().numpy()),2),0,1))
        plt.title('Observed')

        plt.subplot(133)
        plt.imshow(np.clip(np.stack((X_LR[:,:,show[0]].cpu().detach().numpy(),
                             X_LR[:,:,show[1]].cpu().detach().numpy(),
                             X_LR[:,:,show[2]].cpu().detach().numpy()),2),0,1))
        plt.title('f_x')

        plt.subplot(132)
        plt.imshow(np.clip(np.stack((X_Out[:,:,show[0]].cpu().detach().numpy(),
                             X_Out[:,:,show[1]].cpu().detach().numpy(),
                             X_Out[:,:,show[2]].cpu().detach().numpy()),2),0,1))
        plt.title('g_f_x')
        plt.show()