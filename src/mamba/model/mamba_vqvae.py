import torch
import torch.nn as nn
from mamba.model.residual_mamba_layer import ResidualMambaLayer

class MambaVQVAE(nn.Module):
    def __init__(self, 
                 vocab_size=1024,
                 dim=512,
                 state_size=16,
                 num_layers=16,
                 num_attr=3,
                 expansion=2.0,
                 kernel_size=17,
                 dropout_prob=0.1
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_attr = num_attr
        self.dim = dim
        self.dropout_prob = dropout_prob

        # 1. Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.attr_emb = nn.Embedding(3, dim) 
        self.sos_token = nn.Parameter(torch.randn(1, 1, dim))

        # 2. POSITIONAL EMBEDDING 2D LEARNABLE
        self.h_emb = nn.Embedding(16, dim // 2) # Rows
        self.w_emb = nn.Embedding(16, dim // 2) # Cols
        self.attr_pos_emb = nn.Embedding(4, dim)

        # 3. Mamba Layers
        self.layers = nn.ModuleList([
            ResidualMambaLayer(
                dim=dim, 
                state_size=state_size, 
                conv_kernel=kernel_size,
                expansion=expansion
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(dim, vocab_size, bias=False)

        # --- CACHE DEGLI INDICI POSIZIONALI ---
        # 1. Crea griglia Raster
        rows = torch.arange(16).unsqueeze(1).repeat(1, 16) # 0..15 giù
        cols = torch.arange(16).unsqueeze(0).repeat(16, 1) # 0..15 destra
        
        # 2. Stack (256, 2) e aggiungi dimensione batch (1, 256, 2)
        coords = torch.stack([rows.flatten(), cols.flatten()], dim=1).unsqueeze(0)
        
        # 3. Snake Process
        coords_snake = self.snake_process(coords, side_len=16) # (1, 256, 2)
        
        # 4. Rimuovi dim batch (256, 2)
        coords_snake = coords_snake.squeeze(0)
        
        # 5. Register Buffer (Gestione automatica del device)
        self.register_buffer('cached_r_idx', coords_snake[:, 0])
        self.register_buffer('cached_c_idx', coords_snake[:, 1])

    @staticmethod
    def snake_process(indices, side_len=16):
        B, L = indices.shape[:2] # Prende B e L (funziona sia per 2D che 3D tensors)
        
        # Supporta sia (B, L) che (B, L, D) per processare coordinate
        if len(indices.shape) == 3:
            D = indices.shape[2]
            x = indices.clone().view(B, side_len, side_len, D)
            x[:, 1::2] = x[:, 1::2].flip(dims=[2])
            return x.reshape(B, L, D)
        else:
            x = indices.clone().view(B, side_len, side_len)
            x[:, 1::2] = x[:, 1::2].flip(dims=[2])
            return x.reshape(B, L)

    def _get_cached_pos_emb(self):
        """
        Recupera gli embedding usando gli indici cachati.
        """
        # Essendo buffer registrati, self.cached_r_idx è già sul device corretto
        r_emb = self.h_emb(self.cached_r_idx) # (256, dim/2)
        c_emb = self.w_emb(self.cached_c_idx) # (256, dim/2)
        
        # Concatena
        return torch.cat([r_emb, c_emb], dim=-1) # (256, dim)

    def forward(self, idx_seq, attr_seq, training_dropout=True):
        B, L = idx_seq.shape
        device = idx_seq.device

        # --- A. ATTRIBUTE DROPOUT (CFG) ---
        if training_dropout and self.training:
            mask = torch.bernoulli(torch.full((B, 1), self.dropout_prob, device=device)).bool()
            attr_seq = attr_seq.masked_fill(mask, 2)

        # Embeddings
        x_attr = self.attr_emb(attr_seq.long())
        x_img = self.token_emb(idx_seq)
        sos = self.sos_token.expand(B, -1, -1)

        # --- B. Positional Embeddings ---
        # 1. Prompt (SOS + 3 Attr)
        prompt_len = 1 + self.num_attr
        pos_prompt = self.attr_pos_emb(torch.arange(prompt_len, device=device))
        pos_prompt = pos_prompt.unsqueeze(0).expand(B, -1, -1)
        
        # 2. Immagine (Uso cache ottimizzata)
        # Otteniamo (256, dim) -> expand a (B, 256, dim)
        pos_img = self._get_cached_pos_emb().unsqueeze(0).expand(B, -1, -1)

        # Concatenazione
        seq_emb = torch.cat([sos, x_attr, x_img], dim=1)
        full_pos = torch.cat([pos_prompt, pos_img], dim=1)
        
        x = seq_emb + full_pos

        for layer in self.layers:
            x = layer(x)
            
        logits = self.head(x) 
        return logits

    @torch.no_grad()
    def generate(self, attr_seq, steps=256, temperature=1.0, cfg_scale=3.0, top_k=50):
        import torch.nn.functional as F
        
        B = attr_seq.shape[0]
        device = attr_seq.device
        
        # --- PREPARAZIONE DOPPIO BATCH (COND + UNCOND) ---
        null_attrs = torch.full_like(attr_seq, 2)
        combined_attrs = torch.cat([attr_seq, null_attrs], dim=0) # (2B, 3)
        
        # Init inference
        for layer in self.layers:
            layer.inference_start(batch_size=2*B)
            
        # Prompt processing
        sos = self.sos_token.expand(2*B, -1, -1)
        attr = self.attr_emb(combined_attrs.long())
        start_seq = torch.cat([sos, attr], dim=1)
        
        # Positional Prompt
        pos_prompt = self.attr_pos_emb(torch.arange(4, device=device)).unsqueeze(0).expand(2*B, -1, -1)
        start_seq = start_seq + pos_prompt

        curr_state = None
        # Pre-fill (Prompt)
        for i in range(start_seq.shape[1]):
            curr_input = start_seq[:, i:i+1, :]
            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)
            if i == start_seq.shape[1] - 1:
                curr_state = curr_input

        generated_indices = []
        
        # --- OTTIMIZZAZIONE CACHE ---
        # Recuperiamo l'intera mappa posizionale (256, dim) una volta sola
        # ed espandiamo per il batch raddoppiato.
        full_pos_2d = self._get_cached_pos_emb().unsqueeze(0).expand(2*B, -1, -1)
        
        for i in range(steps):
            logits = self.head(curr_state)[:, -1, :]
            
            cond_logits, uncond_logits = torch.split(logits, B, dim=0)
            cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            
            logits_to_sample = cfg_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits_to_sample, min(top_k, logits_to_sample.size(-1)))
                logits_to_sample[logits_to_sample < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits_to_sample, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            
            generated_indices.append(idx)
            
            next_idx = torch.cat([idx, idx], dim=0)
            curr_input = self.token_emb(next_idx)
            
            # Accesso diretto in memoria senza calcoli, già sul device corretto
            curr_pos = full_pos_2d[:, i:i+1, :]
            curr_input = curr_input + curr_pos
            
            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)
            curr_state = curr_input

        return torch.cat(generated_indices, dim=1)