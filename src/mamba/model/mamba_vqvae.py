import torch
import torch.nn as nn
from mamba.model.residual_mamba_layer import ResidualMambaLayer


class MambaVQVAE(nn.Module):
    def __init__(self, 
                 vocab_size=1024, # Dimensione del Codebook del VQ-VAE (quanti possibili token esistono)
                 dim=512,         # Dimensione interna dei vettori (embedding dimension)
                 state_size=16,   # Dimensione dello stato interno SSM di Mamba (h_t)
                 num_layers=16,   # Profondità della rete
                 num_attr=3,      # Numero di attributi/condizioni da passare in input
                 expansion=2.0,   # Fattore di espansione nel blocco Mamba (proietta a dim * expansion)
                 kernel_size=17,  # Kernel della convoluzione locale 1D dentro Mamba
                 dropout_prob=0.1 # Probabilità di droppare gli attributi (per Classifier-Free Guidance)
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_attr = num_attr
        self.dim = dim
        self.dropout_prob = dropout_prob

        # --- 1. EMBEDDINGS (Rappresentazione dei dati) ---
        # Converte gli indici interi (token immagine) in vettori densi.
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Embedding per gli attributi (le classi condizionanti).
        self.attr_emb = nn.Embedding(3, dim) 
        
        # Token speciale "Start Of Sequence". Serve a dire alla rete "Inizia a generare ora".
        # È un parametro apprendibile di dimensione (1, 1, dim).
        self.sos_token = nn.Parameter(torch.randn(1, 1, dim))

        # --- 2. POSITIONAL EMBEDDING 2D LEARNABLE ---
        # Mamba elabora sequenze 1D, ma le immagini sono 2D. 
        # Per preservare l'informazione spaziale, impariamo embedding separati per righe e colonne.
        # Immagine latente attesa: 16x16 token = 256 token totali.
        self.h_emb = nn.Embedding(16, dim // 2) # Embedding per la riga (Y), metà dimensione
        self.w_emb = nn.Embedding(16, dim // 2) # Embedding per la colonna (X), metà dimensione
        # Quando concatenati, h_emb + w_emb formano un vettore di dimensione 'dim'.

        # Positional embedding specifico per la parte iniziale del prompt (SOS + attributi)
        # 4 posizioni: 1 per SOS + 3 per gli attributi.
        self.attr_pos_emb = nn.Embedding(4, dim)

        # --- 3. MAMBA LAYERS (Il "Cervello") ---
        # Una pila di strati Mamba che processano la sequenza.
        self.layers = nn.ModuleList([
            ResidualMambaLayer(
                dim=dim, 
                state_size=state_size, 
                conv_kernel=kernel_size,
                expansion=expansion
            )
            for _ in range(num_layers)
        ])

        # Proiezione finale: trasforma il vettore latente nella probabilità per il prossimo token.
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # --- CACHE DEGLI INDICI POSIZIONALI (Ottimizzazione) ---
        # Qui pre-calcoliamo la griglia delle coordinate per l'immagine 16x16.
        # Invece di ricalcolarla ad ogni forward, la salviamo in memoria.
        
        # 1. Crea griglia Raster (lettura standard riga per riga)
        # rows: [[0,0...], [1,1...], ... [15,15...]]
        rows = torch.arange(16).unsqueeze(1).repeat(1, 16) 
        # cols: [[0,1...15], [0,1...15], ...]
        cols = torch.arange(16).unsqueeze(0).repeat(16, 1) 
        
        # 2. Stack e Flatten: Ogni punto ha coordinata (y, x).
        # (16, 16, 2) -> (256, 2) -> aggiungi batch (1, 256, 2)
        coords = torch.stack([rows.flatten(), cols.flatten()], dim=1).unsqueeze(0)
        
        # 3. SNAKE PROCESS (Punto cruciale per Mamba su Immagini)
        # Se leggiamo un'immagine riga per riga ("raster scan"), quando finiamo la riga 0 (dx)
        # e saltiamo alla riga 1 (sx), c'è un "salto" spaziale enorme nell'immagine, 
        # ma sono adiacenti nella sequenza 1D. Questo confonde i modelli sequenziali.
        # Lo "Snake Process" fa zig-zag: riga 0 (sx->dx), riga 1 (dx->sx), riga 2 (sx->dx).
        # Così manteniamo sempre la continuità spaziale locale.
        coords_snake = self.snake_process(coords, side_len=16) # (1, 256, 2)
        
        # 4. Rimuovi dim batch fittizia -> (256, 2)
        coords_snake = coords_snake.squeeze(0)
        
        # 5. Register Buffer
        # Dice a PyTorch: "Questi sono tensori parte dello stato del modello, salvali (save_dict),
        # muovili su GPU insieme al modello (.cuda()), ma NON calcolare gradienti su di essi".
        self.register_buffer('cached_r_idx', coords_snake[:, 0]) # Indici riga (Y)
        self.register_buffer('cached_c_idx', coords_snake[:, 1]) # Indici colonna (X)

    @staticmethod
    def snake_process(indices, side_len=16):
        """
        Trasforma una sequenza raster (lettura a Z) in una a serpente (zig-zag).
        Input: (Batch, Len, ...)
        """
        B, L = indices.shape[:2] 
        
        # Gestisce sia input solo indici (B, L) che coordinate (B, L, D)
        if len(indices.shape) == 3:
            D = indices.shape[2]
            # Reshape a griglia 2D: (B, 16, 16, D)
            x = indices.clone().view(B, side_len, side_len, D)
            # x[:, 1::2] seleziona le righe dispari (indice 1, 3, 5...).
            # .flip(dims=[2]) inverte l'ordine degli elementi in quelle righe (da dx a sx).
            x[:, 1::2] = x[:, 1::2].flip(dims=[2])
            # Ritorna alla forma piatta sequenziale
            return x.reshape(B, L, D)
        else:
            # Stessa logica per tensori senza ultima dimensione
            x = indices.clone().view(B, side_len, side_len)
            x[:, 1::2] = x[:, 1::2].flip(dims=[2])
            return x.reshape(B, L)

    def _get_cached_pos_emb(self):
        """
        Recupera gli embedding posizionali usando gli indici cachati (snake order).
        """
        # self.cached_r_idx contiene l'ordine Y corretto per lo zig-zag
        # self.cached_c_idx contiene l'ordine X corretto per lo zig-zag
        
        # Lookup negli embedding: (256) -> (256, dim/2)
        r_emb = self.h_emb(self.cached_r_idx) 
        c_emb = self.w_emb(self.cached_c_idx) 
        
        # Concatena: (256, dim/2) + (256, dim/2) -> (256, dim)
        # Ora abbiamo un vettore posizionale unico per ogni pixel, che sa dove si trova
        # nella griglia 2D, ma ordinato come una sequenza 1D a serpente.
        return torch.cat([r_emb, c_emb], dim=-1) 

    def forward(self, idx_seq, attr_seq, training_dropout=True):
        """
        Training Step.
        idx_seq: (B, 256) -> I token dell'immagine (dal VQ-VAE)
        attr_seq: (B, 3) -> Gli attributi di condizione
        """
        B, L = idx_seq.shape
        device = idx_seq.device

        # --- A. ATTRIBUTE DROPOUT (Classifier-Free Guidance - CFG) ---
        # Durante il training, a volte "oscuriamo" la condizione (attributi) sostituendola
        # con un token speciale (indice 2). 
        # Questo insegna al modello a generare immagini sia "condizionate" (seguendo gli attributi)
        # sia "incondizionate" (generando un'immagine generica).
        # In inferenza, useremo la differenza tra le due per guidare la generazione (CFG).
        if training_dropout and self.training:
            # torch.full((B, 1), self.dropout_prob) crea una matrice (B, 1) con tutti valori uguali a dropout_prob
            # torch.bernoulli campiona da una distribuzione di Bernoulli con quella probabilità.
            # Per ogni riga del batch, "lancia una moneta" sbilanciata. 
            # Restituisce 1 (True) con probabilità dropout_prob e 0 (False) con probabilità 1 - dropout_prob.
            mask = torch.bernoulli(torch.full((B, 1), self.dropout_prob, device=device)).bool()
            # Qui il codice prende la sequenza degli attributi e, dove la maschera è True, 
            # sovrascrive il valore originale con il numero 2.
            attr_seq = attr_seq.masked_fill(mask, 2)

        # --- B. PREPARAZIONE SEQUENZA ---
        # 1. Embeddings
        x_attr = self.attr_emb(attr_seq.long()) # (B, 3, dim)
        x_img = self.token_emb(idx_seq)         # (B, 256, dim)
        sos = self.sos_token.expand(B, -1, -1)  # (B, 1, dim) - Espande il parametro singolo per il batch

        # 2. Positional Embeddings
        # Parte PROMPT (SOS + 3 Attributi = 4 token)
        # Assegniamo posizioni 0, 1, 2, 3
        prompt_len = 1 + self.num_attr
        pos_prompt = self.attr_pos_emb(torch.arange(prompt_len, device=device)) # (4, dim)
        pos_prompt = pos_prompt.unsqueeze(0).expand(B, -1, -1) # (B, 4, dim)
        
        # Parte IMMAGINE (256 token)
        # Recuperiamo la griglia "snake" pre-calcolata.
        # (256, dim) -> expand a (B, 256, dim)
        pos_img = self._get_cached_pos_emb().unsqueeze(0).expand(B, -1, -1)

        # 3. Concatenazione finale
        # Sequenza totale: [SOS] -> [Attr1, Attr2, Attr3] -> [Img_Token_0 ... Img_Token_255]
        # Lunghezza totale: 1 + 3 + 256 = 260 token
        seq_emb = torch.cat([sos, x_attr, x_img], dim=1)      # Contenuto
        full_pos = torch.cat([pos_prompt, pos_img], dim=1)    # Posizione
        
        # Somma contenuto + posizione (standard nei Transformer/Mamba)
        x = seq_emb + full_pos

        # --- C. PASSAGGIO NELLA RETE ---
        for layer in self.layers:
            x = layer(x) # Elaborazione Mamba
            
        # Calcolo Logits (predizione del prossimo token)
        logits = self.head(x) 
        return logits

    @torch.no_grad()
    def generate(self, attr_seq, steps=256, temperature=1.0, cfg_scale=3.0, top_k=50):
        """
        Generazione Immagine (Inferenza).
        Utilizza Classifier-Free Guidance (CFG) e KV-Caching (stato Mamba).
        """
        import torch.nn.functional as F
        
        B = attr_seq.shape[0]
        device = attr_seq.device
        
        # --- 1. PREPARAZIONE DOPPIO BATCH (CFG) ---
        # Per applicare la CFG, dobbiamo eseguire il modello due volte in parallelo:
        # 1. Condizionato (con gli attributi reali)
        # 2. Incondizionato (con attributi "nulli" / index 2)
        # Concateniamo tutto in un batch di dimensione 2*B per efficienza.
        null_attrs = torch.full_like(attr_seq, 2)
        combined_attrs = torch.cat([attr_seq, null_attrs], dim=0) # (2B, 3)
        
        # Inizializza lo stato interno dei layer Mamba (cache dello stato h_t).
        # Questo è fondamentale: permette di generare token per token senza riprocessare tutto lo storico.
        for layer in self.layers:
            layer.inference_start(batch_size=2*B)
            
        # --- 2. PREPARAZIONE PROMPT INIZIALE ---
        # Costruiamo la sequenza di start: [SOS] + [Attributi]
        sos = self.sos_token.expand(2*B, -1, -1)
        attr = self.attr_emb(combined_attrs.long())
        start_seq = torch.cat([sos, attr], dim=1) # (2B, 4, dim)
        
        # Aggiungiamo Positional Embeddings (0,1,2,3)
        pos_prompt = self.attr_pos_emb(torch.arange(4, device=device)).unsqueeze(0).expand(2*B, -1, -1)
        start_seq = start_seq + pos_prompt

        # --- 3. PRE-FILL (Processing del Prompt) ---
        # Alimentiamo il prompt nel modello per "caricare" lo stato interno di Mamba.
        # Non generiamo ancora nulla, aggiorniamo solo la memoria del modello.
        curr_state = None
        for i in range(start_seq.shape[1]):
            # Prendiamo un token alla volta (simuliamo arrivo sequenziale)
            curr_input = start_seq[:, i:i+1, :]
            for layer in self.layers:
                # inference_step aggiorna lo stato interno e ritorna l'output
                curr_input = layer.inference_step(curr_input)
            
            # L'ultimo output del prompt sarà l'input per generare il 1° pixel
            if i == start_seq.shape[1] - 1:
                curr_state = curr_input

        generated_indices = []
        
        # --- 4. CACHE POSIZIONALE GENERAZIONE ---
        # Recuperiamo la mappa completa delle posizioni (snake) per i 256 step futuri.
        # expand a (2B, 256, dim)
        full_pos_2d = self._get_cached_pos_emb().unsqueeze(0).expand(2*B, -1, -1)
        
        # --- 5. LOOP AUTOREGRESSIVO ---
        for i in range(steps): # steps = 256 (tutti i pixel dell'immagine)
            # curr_state contiene l'informazione "cosa ho visto finora".
            # La head predice il prossimo token.
            logits = self.head(curr_state)[:, -1, :] # (2B, vocab_size)
            
            # Separazione Condizionato / Incondizionato
            cond_logits, uncond_logits = torch.split(logits, B, dim=0)
            
            # --- FORMULA CLASSIFIER-FREE GUIDANCE ---
            # Output = Uncond + Scale * (Cond - Uncond)
            # Spingiamo la predizione verso la direzione suggerita dalla condizione, allontanandoci da quella generica.
            cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            
            # Sampling: Temperatura (controlla la creatività/casualità)
            logits_to_sample = cfg_logits / temperature
            
            # Sampling: Top-K (taglia la coda lunga delle probabilità basse per evitare artefatti)
            if top_k > 0:
                v, _ = torch.topk(logits_to_sample, min(top_k, logits_to_sample.size(-1)))
                # Maschera tutto ciò che è sotto la soglia k-esima con -infinito
                logits_to_sample[logits_to_sample < v[:, [-1]]] = -float('Inf')
            
            # Calcolo probabilità e scelta casuale (Multinomial)
            probs = F.softmax(logits_to_sample, dim=-1)
            # Invece di prendere sempre e solo il token con la probabilità più alta (metodo chiamato Greedy Search), 
            # il codice usa il campionamento multinomiale.
            # Se il token "A" ha l'80% di probabilità e il token "B" il 20%, la fetta di A sarà molto grande.
            idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            generated_indices.append(idx)
            
            # Preparazione input per il prossimo step
            # Duplichiamo l'indice scelto perché dobbiamo alimentare sia la parte Cond che Uncond
            # con lo STESSO token appena generato (altrimenti le storie divergono).
            next_idx = torch.cat([idx, idx], dim=0) # (2B, 1)
            curr_input = self.token_emb(next_idx)   # Embedding del token
            
            # Aggiunta Positional Embedding
            # Preleviamo il vettore posizionale per lo step 'i' corrente (ottimizzazione cache)
            curr_pos = full_pos_2d[:, i:i+1, :] 
            curr_input = curr_input + curr_pos
            
            # Passaggio nei layer Mamba (aggiorna stato e calcola output)
            for layer in self.layers:
                curr_input = layer.inference_step(curr_input)
            
            # L'output diventa lo stato corrente per la prossima head prediction
            curr_state = curr_input

        # Concatena tutti gli indici generati -> (B, 256)
        return torch.cat(generated_indices, dim=1)