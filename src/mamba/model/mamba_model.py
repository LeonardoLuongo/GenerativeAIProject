import torch
from torch import nn
from residual_mamba_layer import ResidualMambaLayer

EMBEDDING_DIMENSION=512
CONV_KERNEL=4
VOCABULARY_SIZE=len(data.vocabulary)
STATE_SIZE=16
EXPANSION=1.2
MAMBA_LAYERS=6

class MambaModel(nn.Module):
    def __init__(self, vocab_size=VOCABULARY_SIZE,
                       dim=EMBEDDING_DIMENSION,
                       state_size=STATE_SIZE,
                       expansion=EXPANSION,
                       conv_kernel=CONV_KERNEL,
                       num_layers=MAMBA_LAYERS):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, dim)
        self.layers=nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ResidualMambaLayer(dim, state_size, 
                                         conv_kernel, expansion))
        self.output=nn.Linear(dim, vocab_size, bias=False)

    def forward(self, X):
        '''X is a tensor of token indices, with shape (B,L).
           where L is the sequence length and B the batch size.
           The return value is a tensor of logits (i.e. the values
           are NOT normalized with Softmax) with shape (B,L,V), where
           V is the vocabulary size.
        '''
        X=self.embedding(X)
        for layer in self.layers:
            X=layer(X)
        return self.output(X)

    def inference_start(self):
        for layer in self.layers:
            layer.inference_start()

    def inference_step(self, X):
        '''X is a tensor of token indices, with shape (1,1).
           The return value is a tensor of logits (i.e. the values
           are NOT normalized with Softmax) with shape (1,1,V), where
           V is the vocabulary size.
        '''
        X=self.embedding(X)
        for layer in self.layers:
            X=layer.inference_step(X)
        return self.output(X)



def parameter_count(model):
    count=0
    for p in model.parameters():
        count+=p.numel()
    return count


def loss_function(Y,Target):
    (B,L,D)=Y.shape
    Y=Y.reshape(B*L, D)
    Target=Target.reshape(B*L).to(dtype=torch.long)
    return cross_entropy(Y,Target,reduction='mean')
