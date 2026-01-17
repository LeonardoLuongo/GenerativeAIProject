import sys
import os

import torch
from torch import nn
from torch.nn.functional import softplus
from mamba.model.pscan import pscan

# Tenta di importare il kernel ottimizzato
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_KERNEL = True
    print("🚀 Mamba Acceleration: ON (CUDA Kernel attivo)")
except ImportError:
    HAS_CUDA_KERNEL = False
    print("⚠️ Mamba Acceleration: OFF (mamba-ssm non installato o no GPU)")

class MambaBlock(nn.Module):
    def __init__(self, dim, state_size):
        super().__init__()
        self.dim = dim
        self.state_size = state_size

        # --- STESSI PARAMETRI DEL PROF (NON TOCCARE) ---
        # Initialize the A tensor
        A = torch.ones(dim, state_size)
        r = torch.linspace(0.5, state_size/2, state_size)
        A *= r[None, :]
        logA = torch.log(A)
        self.logA = nn.Parameter(logA) # A is -torch.exp(self.logA)

        # define the other parameters and layers
        self.projB = nn.Linear(dim, state_size)
        self.projC = nn.Linear(dim, state_size)
        self.projDelta = nn.Linear(dim, 1, bias=False)
        biasDelta = torch.zeros(1, 1, dim)
        self.biasDelta = nn.Parameter(biasDelta)
        self.softplus = nn.Softplus()
        # -----------------------------------------------

    def forward(self, x):
        # x shape: (B, L, D)
        
        # 1. Calcoliamo le proiezioni usando i layer originali
        #    Così i gradienti aggiornano i pesi 'projB', 'projC', ecc.
        B_proj = self.projB(x)        # (B, L, N)
        C_proj = self.projC(x)        # (B, L, N)
        
        # Calcolo Delta come fa il prof
        Delta = self.softplus(self.biasDelta + self.projDelta(x)) # (B, L, D) o (B, L, 1) a seconda del broadcasting
        
        # Gestione dimensionale di Delta nel caso il prof usi broadcasting strano
        if Delta.shape[-1] == 1:
            Delta = Delta.repeat(1, 1, self.dim) # (B, L, D)

        # Calcolo A come fa il prof
        A = -torch.exp(self.logA)     # (D, N)

        # === STRADA VELOCE (CUDA) ===
        # Il kernel CUDA vuole shape diverse: (B, D, L) invece di (B, L, D)
        u = x.transpose(1, 2)             # (B, D, L)
        delta = Delta.transpose(1, 2)     # (B, D, L)
        B = B_proj.transpose(1, 2)        # (B, N, L)
        C = C_proj.transpose(1, 2)        # (B, N, L)
        
        if delta.dtype != u.dtype:
            delta = delta.to(dtype=u.dtype)

        # Il kernel Selective Scan ufficiale
        # Nota: il codice del prof non ha il parametro D (skip connection dello scan), 
        # quindi passiamo None. z=None perché la gate è fuori da questo blocco.
        y = selective_scan_fn(
            u, delta, A, B, C, 
            D=None, z=None, delta_bias=None, delta_softplus=False
        )
        
        # Rimettiamo a posto le dimensioni: (B, D, L) -> (B, L, D)
        y = y.transpose(1, 2)
            
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