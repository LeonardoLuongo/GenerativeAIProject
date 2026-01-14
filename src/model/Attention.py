import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper per la funzione di attenzione scalata. 
# Utile per mantenere il codice pulito o per switchare tra implementazioni custom 
# e quelle ottimizzate di PyTorch (Flash Attention) in futuro.
def scaled_dot_product_attention(Q, K, V, is_causal=False):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=1, is_causal=False):
        super().__init__()
        
        # La dimensione totale dell'embedding deve essere 
        # divisibile per il numero di teste, altrimenti non possiamo splittare 
        # i canali equamente tra le teste.
        assert dim % heads == 0
        
        self.dim = dim
        self.heads = heads
        self.is_causal = is_causal # Flag per mascheramento (usato es. nei decoder per non guardare il futuro)
        
        # Proiezioni lineari per mappare l'input X negli spazi semantici di
        # Query (cosa cerco), Key (etichetta di contenuto) e Value (contenuto effettivo).
        # Sono matrici apprendibili W_q, W_k, W_v.
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        
        # Proiezione di output (W_o).
        # Serve a "mescolare" le informazioni provenienti dalle diverse teste dopo la concatenazione.
        # Nota implementativa: se c'è una sola testa, la concatenazione è un'identità, 
        # quindi tecnicamente potremmo saltare questa proiezione per risparmiare parametri (scelta di design).
        if heads > 1:
            self.output_projection = nn.Linear(dim, dim)
        else:
            self.output_projection = None

    def forward(self, X):
        # X shape: (Batch_Size, Sequence_Length, Embedding_Dim) -> (N, L, D)
        
        # 1. Proiezione lineare: trasformiamo l'input nei tre vettori Q, K, V
        Q = self.query_projection(X)
        K = self.key_projection(X)
        V = self.value_projection(X)
        
        # 2. Multi-Head Split
        # Se stiamo usando più teste, dobbiamo dividere la dimensione dell'embedding 
        # in 'heads' sottospazi indipendenti per catturare relazioni diverse.
        if self.heads > 1:
            Q = self.split_heads(Q)
            K = self.split_heads(K)
            V = self.split_heads(V)
            # Ora la shape è (N, Heads, L, D_head) pronta per il calcolo parallelo
            
        # 3. Core Attention Mechanism
        # Calcolo di Softmax((QK^T) / sqrt(d_k)) * V
        # Qui avviene l'estrazione delle feature basata sulla similarità tra Q e K.
        Y = scaled_dot_product_attention(Q, K, V, is_causal=self.is_causal)
        
        # 4. Ricomposizione
        # Se abbiamo splittato le teste, ora dobbiamo concatenarle di nuovo 
        # per tornare alla dimensione originale dell'embedding.
        if self.heads > 1:
            Y = self.join_heads(Y)
        
        # 5. Output Projection
        # Applicazione della trasformazione finale W_o per combinare le feature delle varie teste.
        if self.output_projection:
            Y = self.output_projection(Y)
            
        return Y

    def split_heads(self, X):
        """
        Trasforma il tensore per parallelizzare il calcolo sulle teste.
        Input: (N, L, D)
        Output: (N, H, L, D_head)
        """
        (N, L, D) = X.shape
        H = self.heads
        
        # Primo step: Reshape. Separiamo la dimensione D in (H, D/H).
        # Shape diventa: (N, L, H, D_head)
        X = X.reshape(N, L, D // H, H)
        
        # Secondo step: Permute. Spostiamo la dimensione delle teste (H) 
        # prima della lunghezza della sequenza (L). 
        # Questo è CRUCIALE perché PyTorch esegue le operazioni di matrice (matmul) 
        # sulle ultime due dimensioni. Facendo così, 'H' viene trattata come una 
        # dimensione batch aggiuntiva, parallelizzando il calcolo.
        # Shape diventa: (N, H, L, D_head)
        X = X.permute(0, 3, 1, 2)
        
        return X

    def join_heads(self, X):
        """
        Operazione inversa di split_heads.
        Equivale a concatenare le teste lungo l'asse dei canali.
        Input: (N, H, L, D_head)
        Output: (N, L, D)
        """
        (N, H, L, D_head) = X.shape # Qui D è la dimensione della singola testa
        
        # Riportiamo L prima di D_head e mettiamo H in fondo.
        # Shape diventa: (N, L, H, D_head)
        X = X.permute(0, 2, 3, 1)
        
        # Il reshape finale fonde le ultime due dimensioni (H * D_head).
        # Poiché in memoria i dati sono contigui, questo equivale a concatenare 
        # i vettori delle diverse teste.
        # Shape diventa: (N, L, D_originale)
        X = X.reshape(N, L, D_head * H)
        
        return X