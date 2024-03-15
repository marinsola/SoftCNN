import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from math import sqrt,ceil

class MeightWix(nn.Module):
    
    def __init__(self,in_dim,stride=(1,1),bias=True,device=None,initbound=1/32):
        '''
        in_dim : vector input dimension (e.g. 32*32)
        device : device used in training
        stride : tupel containing stride steps
        initbound : bounds for initialisation distribution, the parameters are
                    drawn uniformly from [-initbound, initbound]
        '''
        
        super().__init__()
        self.in_dim=in_dim                 
        self.out_dim=ceil(in_dim/stride[0])
        self.device=device                 
        self.stride=stride                 
        self.initbound = initbound         
                                           
        self.v = nn.Parameter(torch.empty(self.in_dim*self.stride[1],
                                          device=self.device))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_dim,device=self.device))
        self.reset_parameters() 
        
    def reset_parameters(self):
        
        init.uniform_(self.v, a=-self.initbound, b=self.initbound)
        if self.bias is not None:
            init.uniform_(self.bias, -self.initbound, self.initbound)
    
    def forward(self,x):
        fau = torch.tile(self.v.unsqueeze(0),(1,2))
        we =  torch.as_strided(fau, size=(self.out_dim,self.in_dim), 
                               stride=self.stride,storage_offset=0)
        return F.linear(x, we ,self.bias)


class BixMlock(nn.Module):  
    
    def __init__(self,in_channels,out_channels,in_size,stride,device='cpu',
                 initbound=1/32):
        """
        Creates in_channel many MeightWixes per output channel 
        and adds their output into the output channel. For further 
        clarification see the graphic in the report.
    
        Input: Tensor of dimension (batch,in_channels,in_size)
               (eg: (1000,3,32*32) )

        Output: Tensor of dimension (batch,out_channels,ceil(in_size/stride[0]))
                (eg: (1000,20,16*16)) if stride=(4,1))

        Number of parameters: in_size * out_channels * in_channels * stride[1]
        """  
        
        super(BixMlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.stride = stride
        self.initbound = initbound
        
        self.channels =  nn.ModuleList([nn.ModuleList([MeightWix(self.in_size,
                                                                 stride=self.stride,
                                                                 device=device,
                                                                 initbound=self.initbound)
                                                       for i in range(in_channels)]) 
                                        for j in range(out_channels)])
            
    def forward(self,x):
        
        out_features = []
        for j in range(self.out_channels):
            
            features = 0
            for i in range(self.in_channels):
                
                features += self.channels[j][i](torch.flatten(x[:,i,:],1))
                                
            out_features.append(features)
            
        return torch.stack(out_features,dim=1)