
import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils

import torchsde
from torch import nn
import copy
import matplotlib.pyplot as plt

import numpy as np
import pdb
from torchdiffeq import odeint

import pynvml
from Networks.nets import MLP, MLPL, MLPWithSkip,  MLPHypernetwork, LinearNet, LipSwish

import networkx as nx
from scipy.linalg import eigh

def find_params(SML):
    
    params = None
    
    if  SML.strategy =="ENC":
        
        params = list(SML.poly.parameters()) 

   
    elif SML.strategy =="HNC":
        
        params = list(SML.poly.hhnet.weight_generator.parameters()) 
    
    elif SML.strategy=="tHNC":

        params = list(SML.poly.hhnet.weight_generator.parameters())
    return params
    

class HNC_strategy(nn.Module):
    def __init__(self, skip, dim, layers,batch, device):
        super().__init__()
        self.batch=batch
        self.dim=dim
      
      
        h_layers = (2+dim,)*layers+(8,)
            
        self.polyf = MLP(in_size=2*dim, out_size=dim,mlp_size=dim+1,num_layers=1)        
        self.hhnet = MLPHypernetwork( hyper_dims = 1+dim, target_network = self.polyf, layers=h_layers, hyperfan=True)
     
        self.device=device

    def forward(self, t, y,freqs):

        
        ts=torch.full((1, 1), t).to(self.device)
        tss=ts.repeat(self.batch,1)
  
        generated_vmap_params= torch.vmap(self.hhnet.generate_params)( torch.cat([tss-0.5,freqs],dim=1))
        xtr=torch.cos(y[:,0:self.dim])
        ytr=torch.sin(y[:,0:self.dim])
        input=torch.cat([xtr,ytr],axis=1)
       
        cforce = torch.vmap(self.hhnet.forward)(input,generated_vmap_params )

        return cforce 


class tHNC_strategy(nn.Module):
    def __init__(self, skip, dim, layers,batch, device):
        super().__init__()
        self.batch=batch
        self.dim=dim
      
      
        h_layers = (dim,)*layers+(1,)
            
        self.polyf = MLP(in_size=3*dim, out_size=dim,mlp_size=2*dim+1,num_layers=1)        
        self.hhnet = MLPHypernetwork( hyper_dims = 1, target_network = self.polyf, layers=h_layers, hyperfan=True)
     
        self.device=device

    def forward(self, t, y,freqs):

        
        ts=torch.full((1, 1), t).to(self.device)
        tss=ts.repeat(self.batch,1)
        generated_vmap_params= torch.vmap(self.hhnet.generate_params)(tss)
        #generated_vmap_params= torch.vmap(self.hhnet.generate_params)( torch.cat([tss-0.5,freqs],dim=1))
        xtr=torch.cos(y[:,0:self.dim])
        ytr=torch.sin(y[:,0:self.dim])
        input=torch.cat([xtr,ytr,freqs],axis=1)
       
        cforce = torch.vmap(self.hhnet.forward)(input,generated_vmap_params )

        return cforce 



class ENC_strategy(nn.Module):
    def __init__(self, skip, dim, layers,batch, device, emb_layers=0):
        super().__init__()
        self.batch=batch
        self.dim=dim
      
       
        self.poly = MLP(in_size=1+3*dim, out_size=dim,mlp_size=(2+3*dim),num_layers=1+layers)   
        
        if emb_layers==0:
            self.embed=torch.nn.Identity()
        else:
            self.embed = MLP(in_size=1+dim, out_size=1+dim, mlp_size=2+dim, num_layers=1+emb_layers)
        self.device=device
 
    def forward(self, t, y,freqs):

        
        ts=torch.full((1, 1), t).to(self.device)
        tss=ts.repeat(self.batch,1)
        xtr=torch.cos(y[:,0:self.dim])
        ytr=torch.sin(y[:,0:self.dim])
        emb_input=torch.cat([tss-0.5,freqs],axis=1)
        emb= self.embed(emb_input)
        input=torch.cat([xtr,ytr,emb],axis=1)  
        
        cforce=self.poly(input)
     
        return cforce 

class KuramotoHyperUniversal(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, batch, dim, K, A,  layers, skip, strategy, device, emb=0, freqs=None):
        super(KuramotoHyperUniversal, self).__init__()
        self.batch=batch
        self.dim=dim
        self.K=K
        self.freqs=freqs

        self.device=device
        self.A=A
      
        self.strategy=strategy


        if strategy=="HNC":

            self.poly= HNC_strategy(skip, dim, layers,batch, device).to(device)
        
        elif strategy=="ENC":

            self.poly= ENC_strategy(skip, dim, layers,batch,  device,emb).to(device)

        elif strategy=="tHNC":

            self.poly= tHNC_strategy(skip, dim, layers,batch,  device).to(device)
    def f(self, t, y):
       
        
        cforce=self.poly(t-0.5,y,self.freqs)
            
        ymat=y[:,0:self.dim].unsqueeze(2)
     
        ytr=ymat.repeat(1,1,self.dim)
        ytr1=torch.transpose(ytr, 2,1)
   
        phasematrix=ytr1-ytr
    
        forcematrix=torch.sin(phasematrix)*self.A
        
        force=self.K*cforce*torch.sum(forcematrix,2)/self.dim

        force+=self.freqs

        f1=(torch.sum((cforce)**2,dim=1)).unsqueeze(1)
        return torch.cat([force,f1],dim=1)

    def g(self, t, y):
        
        noise=0.0001*torch.ones_like(y[:,0:self.dim]) 
        zero_pad=0*y[:,0].unsqueeze(1)
     
        return torch.cat([noise,zero_pad],dim=1) 


