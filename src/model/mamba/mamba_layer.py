import torch
from torch import nn
from model.mamba.mamba_block import MambaBlock


class MambaLayer(nn.Module):
    def __init__(self, dim, state_size, conv_kernel=4,
                 expansion=1):
        super().__init__()
        edim=int(dim*expansion)
        self.dim=dim
        self.edim=edim
        self.state_size=state_size
        self.conv_kernel=conv_kernel

        self.activation=nn.SiLU()
        self.proj_1=nn.Linear(dim, edim)
        self.conv=nn.Conv1d(edim, edim, conv_kernel, padding=conv_kernel-1)
        self.mamba=MambaBlock(edim, state_size)
        self.proj_2=nn.Linear(dim, edim)
        self.proj_3=nn.Linear(edim, dim)

    def forward(self, x):
        ex=self.proj_1(x)
        ex=self.do_conv(ex)
        ex=self.activation(ex)
        y=self.mamba(ex)
        g=self.proj_2(x)
        g=self.activation(g)
        y*=g
        y=self.proj_3(y)
        return y


    @torch.no_grad
    def inference_start(self, batch_size=1):
        self.mamba.inference_start(batch_size)
        self.cached_x=torch.zeros(batch_size, self.conv_kernel, self.edim)

    @torch.no_grad
    def inference_step(self, x):
        if self.cached_x.device!=x.device:
            self.cached_x=self.cached_x.to(device=x.device)

        ex=self.proj_1(x)
        ex=self.do_conv(ex, True)
        ex=self.activation(ex)
        y=self.mamba.inference_step(ex)
        g=self.proj_2(x)
        g=self.activation(g)
        y*=g
        y=self.proj_3(y)
        return y

        
    def do_conv(self, x, inference=False):
        if inference:
            ck=self.conv_kernel
            ed=self.edim
            self.cached_x=torch.cat([self.cached_x, x], dim=1)[:,1:]
            x=self.cached_x
        L=x.shape[1] 
        x=x.permute(0, 2, 1)
        y=self.conv(x)[:,:,:L]
        y=y.permute(0, 2, 1)
        if inference:
            y=y[:,-1:,:]
        return y

