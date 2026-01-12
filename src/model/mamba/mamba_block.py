import torch
from torch import nn
from model.mamba.pscan import pscan


class MambaBlock(nn.Module):
    def __init__(self, dim, state_size):
        super().__init__()
        self.dim=dim
        self.state_size=state_size

        # Initialize the A tensor
        A=torch.ones(dim, state_size)
        r=torch.linspace(0.5, state_size/2, state_size)
        A*=r[None,:]
        logA=torch.log(A)
        self.logA=nn.Parameter(logA) # A is -torch.exp(self.logA)

        # define the other parameters and layers
        self.projB=nn.Linear(dim, state_size)
        self.projC=nn.Linear(dim, state_size)
        self.projDelta=nn.Linear(dim, 1, bias=False)
        biasDelta=torch.zeros(1,1,dim)
        self.biasDelta=nn.Parameter(biasDelta)
        self.softplus=nn.Softplus()

    def forward(self, x):
        self.clean_cached()
        A=self.computeA()
        B,C,Delta=self.computeBCDelta(x)
        Abar, Bbar=self.discretize(Delta, A, B)
        h=self.perform_scan(Abar, Bbar, x)
        y=torch.einsum('bln,bldn->bld', C, h)
        return y

    @torch.no_grad
    def inference_start(self, batch_size=1):
        self.cached_A=self.computeA()
        self.cached_h=torch.zeros(
                batch_size, 1, self.dim, self.state_size,
                device=self.cached_A.device)

    def clean_cached(self):
        self.cached_A=None
        self.cached_h=None
        

    @torch.no_grad
    def inference_step(self, x):
        A=self.cached_A
        B,C,Delta=self.computeBCDelta(x)
        Abar, Bbar=self.discretize(Delta, A, B)
        h=Abar*self.cached_h + Bbar*x[...,None]
        y=torch.einsum('bln,bldn->bld', C, h)
        self.cached_h.copy_(h)
        return y

    def computeA(self):
        return -torch.exp(self.logA)

    def computeBCDelta(self, x):
        B=self.projB(x)
        C=self.projC(x)
        Delta=self.softplus(self.biasDelta+self.projDelta(x))
        return B,C,Delta

    def discretize(self, Delta, A, B):
        DeltaA=Delta[:,:,:,None]*A[None,None,:,:]
        Abar=torch.exp(DeltaA)
        DeltaB=Delta[:,:,:,None]*B[:,:,None,:]
        Bbar=(Abar-1.0)/(DeltaA + 1e-7)*DeltaB
        return Abar, Bbar

    def perform_scan(self, Abar, Bbar, x): 
        Atilde=Abar
        Xtilde=Bbar*x[...,None]
        return pscan(Atilde, Xtilde)