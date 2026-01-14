import torch
import torch.nn as nn
from model.mamba.residual_mamba_layer import ResidualMambaLayer

class MambaVQVAE(nn.Module):
    def __init__(self, 
                 vocab_size=1024,   # Dimensione codebook VQ-VAE
                 dim=512,           # Deve matchare la dim del ResidualMambaLayer
                 state_size=16,
                 num_layers=6,
                 num_attr=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_attr = num_attr
        self.dim = dim

        # 1. Embedding per i token del VQ-VAE (Indici 0-1023)
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # 2. Embedding per gli attributi (0 o 1)
        # Usiamo 2 vettori separati per ogni attributo per dare info posizionale implicita
        self.attr_emb = nn.Embedding(2, dim) 
        
        # 3. Token speciale di inizio sequenza (SOS)
        # Serve per dire a Mamba "Inizia a generare basandoti sugli attributi"
        self.sos_token = nn.Parameter(torch.randn(1, 1, dim))

        # 4. Positional Embedding (Opzionale ma aiuta molto su sequenze corte come 16x16)
        self.pos_emb = nn.Embedding(256 + num_attr + 1, dim)

        # 5. Stack di Mamba (Codice del PROF)
        self.layers = nn.ModuleList([
            ResidualMambaLayer(dim=dim, state_size=state_size, conv_kernel=4, expansion=1.1)
            for _ in range(num_layers)
        ])

        # 6. Output Head (predice il prossimo indice del codebook)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx_seq, attr_seq):
        """
        idx_seq: (B, 256) -> Indici dal VQ-VAE
        attr_seq: (B, 3)  -> Attributi (0 o 1)
        """
        B, L = idx_seq.shape
        device = idx_seq.device

        # A. Embeddings
        # Attributi: (B, 3) -> (B, 3, D)
        x_attr = self.attr_emb(attr_seq.long())
        
        # Immagine: (B, 256) -> (B, 256, D)
        x_img = self.token_emb(idx_seq)
        
        # SOS Token: (1, 1, D) -> (B, 1, D)
        sos = self.sos_token.expand(B, -1, -1)

        # B. Costruiamo la sequenza completa: [SOS, ATTR1, ATTR2, ATTR3, IMG_TOKENS...]
        # Input sequence (per Teacher Forcing escludiamo l'ultimo token immagine)
        # Sequence layout: [SOS, Attr(3), Img(0...254)] -> Predice -> [Attr(3), Img(0...255)]
        
        # Concateniamo tutto: (B, 1+3+256, D)
        full_seq_emb = torch.cat([sos, x_attr, x_img], dim=1)
        
        # Aggiungiamo Positional Encoding
        seq_len = full_seq_emb.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        full_seq_emb = full_seq_emb + self.pos_emb(positions)

        # C. Passaggio Mamba
        x = full_seq_emb
        for layer in self.layers:
            x = layer(x)
            
        # D. Output
        # Ci interessa predire i token immagine.
        # L'output corrispondente a SOS predice il primo attributo (non ci serve)
        # ...
        # L'output corrispondente all'ultimo Attributo predice il Primo Pixel (ci serve)
        # Prendiamo solo la parte che predice l'immagine
        
        # Input agli strati: [SOS, A1, A2, A3, P0, P1 ... P254]
        # Output layers:     [h0,  h1, h2, h3, h4, h5 ... h258]
        # h3 (dopo A3) deve predire P0. h258 (dopo P254) deve predire P255.
        
        # Quindi prendiamo l'output dagli indici 4 in poi (saltiamo SOS + 3 Attr)
        # Nota: dobbiamo assicurarci di avere la lunghezza giusta per il confronto
        
        logits = self.head(x) 
        
        # Ritorniamo logits. Taglieremo in fase di loss calculation
        return logits

    @torch.no_grad()
    def generate(self, attr_seq, steps=256, temperature=1.0, top_k=50):
        """
        Generazione con Multinomial Sampling.
        temperature: >1.0 più caotico/diverso, <1.0 più conservativo/stabile.
        top_k: considera solo i k token più probabili (taglia via il rumore).
        """
        import torch.nn.functional as F
        
        B = attr_seq.shape[0]
        device = attr_seq.device
        
        # 1. Start Inference del Prof
        for layer in self.layers:
            layer.inference_start(batch_size=B)
            
        # 2. Processiamo il "Prompt" (SOS + Attributi)
        sos = self.sos_token.expand(B, -1, -1)
        attr = self.attr_emb(attr_seq.long())
        
        # Sequenza iniziale: [SOS, Attr1, Attr2, Attr3]
        start_seq = torch.cat([sos, attr], dim=1)
        
        # Positional embedding per la parte iniziale
        positions = torch.arange(4, device=device).unsqueeze(0)
        start_seq = start_seq + self.pos_emb(positions)

        # Alimentiamo Mamba con il contesto iniziale
        next_token_embed = None
        for i in range(start_seq.shape[1]):
            curr_input = start_seq[:, i:i+1, :]
            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)
            
            if i == start_seq.shape[1] - 1:
                next_token_embed = curr_input

        # 3. Generazione Loop Immagine
        generated_indices = []
        curr_state = next_token_embed
        
        for i in range(steps):
            # Ottieni i Logits dall'ultimo stato
            logits = self.head(curr_state) # (B, 1, Vocab)
            logits = logits[:, -1, :]      # (B, Vocab)
            
            # --- MULTINOMIAL SAMPLING ---
            # 1. Applica Temperatura
            logits = logits / temperature
            
            # 2. Top-K Filtering (Opzionale ma raccomandato)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Metti a -inf tutto ciò che è sotto la soglia del k-esimo elemento
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 3. Calcola Probabilità e Campiona
            probs = F.softmax(logits, dim=-1)
            
            # Qui avviene la magia: peschiamo in base alla probabilità, non al massimo
            idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            generated_indices.append(idx)
            
            # --- PREPARAZIONE PROSSIMO STEP ---
            # Embed del token appena predetto
            curr_input = self.token_emb(idx)
            
            # Aggiungi posizione (offset di 4 perché abbiamo SOS+3Attr prima)
            pos_idx = torch.tensor([4 + i], device=device)
            curr_input = curr_input + self.pos_emb(pos_idx)
            
            # Mamba step
            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)
            curr_state = curr_input

        return torch.cat(generated_indices, dim=1)