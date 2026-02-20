import torch
import torchsde
from torch import nn
import numpy as np
import pdb
from Networks.nets import MLP, MLPL, MLPWithSkip, MLPHypernetwork, LinearNet, StableResNODEMLP
from hypernn.torch import TorchHyperNetwork

def find_params(SML):
    
    params = None
    
    if SML.strategy == "HNC":
        
        params = SML.poly.hhnet.weight_generator.parameters()

    elif SML.strategy == "ENC":

        params = SML.poly.polyf.parameters()
        
    return params 



class HNC_strategy(nn.Module):
    def __init__(self,dim,outputdim, layers,batch,device, skip, amat=None):
        super().__init__()
        self.batch=batch
        self.dim=dim
        self.amat=amat
     
        self.device=device
       
        h_layers = (2+self.dim,)*layers
     
        self.polyf = LinearNet(in_size=self.dim, out_size=outputdim)

        self.hhnet = MLPHypernetwork( hyper_dims = 1+self.dim, target_network = self.polyf, layers=h_layers, hyperfan=True, skip=skip)

    def forward(self, t, y):
        
        ts=torch.full((self.batch, 1), t).to(self.device)
     
        hyper_inp=torch.cat((ts-0.5,self.amat),dim=1)
   
        generated_vmap_params= torch.vmap(self.hhnet.generate_params)(hyper_inp)

        cforce = torch.vmap(self.hhnet.forward)(y[:,0:self.dim],generated_vmap_params )
        return cforce  
        
class ENC_strategy(nn.Module):
    def __init__(self, dim,outputdim, layers,batch,device, skip, amat=None):
        super().__init__()
        self.batch=batch
        self.dim=dim
        self.amat=amat
     
        self.device = device
        if skip:
            self.polyf = StableResNODEMLP(in_size=1+2*self.dim,out_size=outputdim,mlp_size=2+2*self.dim,num_layers=layers)  
        else:
            self.polyf = MLP(in_size=1+2*self.dim,out_size=outputdim,mlp_size=2+2*self.dim,num_layers=layers)  
        
    def forward(self, t, y):
        ts=torch.full((self.batch, 1), t).to(self.device)
        
        cforce=self.polyf(torch.cat((y[:,0:self.dim],self.amat,ts-0.5),dim=1))
        
        return cforce  
    



class HyperCoeffsLinearControlStochasticLQRImpl(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, A, B, batch_size,  device, hlayers, strategy, skip=False):
     
        super(HyperCoeffsLinearControlStochasticLQRImpl, self).__init__()
        self.batch=batch_size
        self.At = A.to(device)
        self.Bt = B.to(device)
       
        self.strategy= strategy
        self.dim=A.size(0)
        outputdim= B.size(1)
     

        hlayers=hlayers

        if strategy == "HNC":
            
            self.poly = HNC_strategy(self.dim,outputdim, hlayers,batch_size,device,skip)
        
        elif strategy == "ENC":

            self.poly = ENC_strategy(self.dim,outputdim, hlayers,batch_size,device, skip)
   
    def f(self, t, y):

        forces = self.poly(t,y)

        f = torch.einsum('kji,ki->kj', self.At, y[:,0:self.dim]) + torch.einsum('ji,ki->kj', self.Bt, forces)
     
        f2 = (torch.sum((forces**2),dim=1)).unsqueeze(1)
       
        return torch.cat([f,f2],dim=1)

    def g(self, t, y):
        
        noise=0.0*torch.ones_like(y[:,0:self.dim]) 
        zero_pad=0*y[:,0].unsqueeze(1)
       
        return torch.cat([noise,zero_pad],dim=1)


