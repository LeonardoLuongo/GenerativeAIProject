import torch
from torch import nn
import torch.nn.functional as F

# --- BLOCCO RESIDUALE ---
# Rimedia al problema del vanishing gradient.
# L'input viene sommato all'output delle convoluzioni (skip connection).
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            # Mantiene le dimensioni spaziali (padding=1 con kernel 3x3)
            nn.Conv2d(dim, dim, 3, padding=1),
            # Normalizzazione per stabilità
            # Divide i canali in gruppi per la normalizzazione
            nn.GroupNorm(32, dim),
            # Attivazione non lineare f(x)=max(alpha*x,x)
            # Evita il problema dei neuroni morti rispetto a ReLU
            nn.LeakyReLU(), 
            # Mantiene le dimensioni spaziali (padding=1 con kernel 3x3)
            nn.Conv2d(dim, dim, 3, padding=1), 
            # Normalizzazione per stabilità
            # Divide i canali in gruppi per la normalizzazione
            nn.GroupNorm(32, dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

# --- QUANTIZZATORE VETTORIALE (Versione EMA - High Performance) ---
# Questa classe converte i vettori continui dell'encoder nel vettore più vicino
# presente nel "codebook" (il dizionario dei token).
# Usa EMA (Exponential Moving Average) per aggiornare il dizionario, che è più stabile
# rispetto alla backpropagation standard per questo tipo di architettura.
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        # Dimensione vocabolario (es. 1024)
        self._num_embeddings = num_embeddings 
        # Quanto "impegnare" l'encoder a stare vicino al codebook
        self._commitment_cost = commitment_cost

        # Parametri per l'aggiornamento EMA (Media Mobile Esponenziale)
        self.decay = decay
        self.epsilon = epsilon
        # Inizializzazione del Codebook (Dizionario)
        embedding = torch.randn(self._num_embeddings, self._embedding_dim)

        # register_buffer significa che sono tensori salvati nel modello ma non aggiornati dal gradiente (optimizer)
        # Li aggiorneremo noi con la logica EMA.
        self.register_buffer('_embedding', embedding)

        # Se un codice viene usato 10 volte nel Batch A, ma 0 volte nel Batch B, 
        # la sua "importanza" non dovrebbe crollare a zero istantaneamente.
        # _ema_cluster_size serve a mantenere un conteggio cumulativo e smussato 
        # di quanto ogni vettore del codebook sia stato utilizzato nel tempo.
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', embedding.clone())

    def forward(self, inputs):
        # Gestione Dimensioni
        # Spostiamo i canali alla fine: (Batch, C, H, W) -> (Batch, H, W, C)
        # .contiguous() riordina i dati in memoria per evitare errori successivi con view()
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Appiattiamo l'immagine: invece di avere righe e colonne, mettiamo
        # tutti i pixel in fila indiana. Ogni pixel è un vettore di C dimensioni.
        # flat_input diventa una matrice (Numero_Totale_Pixel, Embedding_Dim)
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # CALCOLO DISTANZE
        # Iteriamo su ogni vettore del codebook (dizionario) e calcoliamo la distanza
        # da tutti i pixel di input.
        distances_list = []
        
        for i in range(self._num_embeddings):
            # Prendiamo l'i-esimo vettore del codebook
            codebook_vector = self._embedding[i]
            
            # Calcoliamo la Distanza Euclidea Quadrata: (input - codebook)^2
            # La sottrazione avviene pixel per pixel grazie al broadcasting
            diff = flat_input - codebook_vector
            dist_squared = torch.sum(diff * diff, dim=1) # Somma dei quadrati
            
            distances_list.append(dist_squared)
            
        # Uniamo tutte le distanze calcolate in un'unica matrice
        # Shape: (Numero_Pixel, Num_Embeddings)
        distances = torch.stack(distances_list, dim=1)

        # TROVA L'INDICE MIGLIORE
        # Per ogni pixel, troviamo l'indice della colonna con la distanza minore.
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # ONE-HOT ENCODING
        # Trasformiamo gli indici in vettori di zeri con un 1 nella posizione dell'indice.
        # Es: tabella con righe=Numero_Pixel e colonne=Num_Embeddings
        # pix1: [0, 0, 1, 0, 0, ...]  (ha scelto il codice 2)
        # pix2: [1, 0, 0, 0, 0, ...]  (ha scelto il codice 0)
        # ...
        encodings = F.one_hot(encoding_indices.squeeze(), self._num_embeddings).float()

        # AGGIORNAMENTO CODEBOOK
        if self.training:
            # Contiamo quante volte è stato usato ogni vettore del codebook
            # Sommiamo lungo le colonne della matrice one-hot
            current_counts = torch.sum(encodings, dim=0)

            # Aggiornamento EMA della "cluster size" (frequenza d'uso)
            # Nuovo = Vecchio * Decay + Corrente * (1 - Decay)
            # Se un codice viene usato 10 volte nel Batch A, ma 0 volte nel Batch B, 
            # la sua "importanza" non dovrebbe crollare a zero istantaneamente.
            # updated_cluster_size serve a mantenere un conteggio cumulativo e smussato 
            # di quanto ogni vettore del codebook sia stato utilizzato nel tempo.
            updated_cluster_size = (self._ema_cluster_size * self.decay) + (current_counts * (1 - self.decay))
            
            # Salviamo il risultato (senza rompere il grafo)
            # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
            with torch.no_grad():
                self._ema_cluster_size.copy_(updated_cluster_size)
            
            # Calcoliamo la somma dei vettori di input assegnati a ogni codice
            # "Per ogni codice K, somma tutti i pixel che sono stati mappati a K"
            #   encodings.t() = [ 
            #       [1, 0, 1], <-- Codice 0 usato da pixel 0 e 2
            #       [0, 1, 0], <-- Codice 1 usato da pixel 1
            #   ]
            #   x
            #  [
            #       [1, 1],  <-- Pixel 0
            #       [2, 2],  <-- Pixel 1
            #       [3, 3],  <-- Pixel 2
            #   ]
            # =
            #  [
            #       [[1,1]+[3, 3]],  <-- Somma vettori per codice 0
            #       [2, 2],  <-- Somma vettori per codice 1
            #   ]
            #
            dw_embed = torch.matmul(encodings.t(), flat_input)

            # Aggiornamento EMA dei vettori del codebook (i pesi veri e propri)
            updated_w = (self._ema_w * self.decay) + (dw_embed * (1 - self.decay))
            with torch.no_grad():
                self._ema_w.copy_(updated_w)
            
            # Normalizzazione e Assegnazione Finale
            # Calcoliamo la somma totale per normalizzare
            n = torch.sum(self._ema_cluster_size.data)
            
            # aggiungiamo epsilon per evitare divisioni per zero
            # nel caso in cui un vettore del codebook non venisse mai scelto da nessun pixel
            cluster_size_smoothed = (
                (self._ema_cluster_size + self.epsilon) / 
                (n + self._num_embeddings * self.epsilon) * n
            )
            
            # Calcolo finale: Peso = Somma_Vettori / Numero_Utilizzi
            # RuntimeError: The size of tensor a (512) must match the size of tensor b (3) at non-singleton dimension 1
            # prima di unsqueezze(1): [v1, v2, v3]
            # dopo unsqueeze(1): [[v1], [v2], [v3]]
            normalized_embedding = self._ema_w / cluster_size_smoothed.unsqueeze(1)
            with torch.no_grad():   
                self._embedding.copy_(normalized_embedding)

        # CALCOLO METRICHE (Perplessità)
        # La perplessità è un numero che ci dice quanti dei nostri vettori nel codebook vengono effettivamente scelti dalla rete.
        # Valori più alti indicano un uso più vario del codebook, valori bassi indicano che pochi vettori vengono usati spesso.
        # perplexity = exp(- Σ p(x) log(p(x))) = exp(entropia)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # QUANTIZZAZIONE E COMMITMENT LOSS
        # Sostituiamo i pixel originali con quelli del codebook
        quantized = F.embedding(encoding_indices, self._embedding).view(input_shape)
        
        # Loss di "impegno": vogliamo che l'encoder non si allontani troppo dal codebook
        # Questa loss forza l'encoder a produrre vettori che siano vicini ai prototipi esistenti nel codebook.
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator:
        # Copiamo i valori quantizzati, ma lasciamo passare i gradienti dell'input originale
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encoding_indices


    def restart_unused_codes(self, inputs, threshold=1.0):
        # Preparazione Input
        # Come nel forward, appiattiamo l'immagine in una lista di vettori (pixel)
        # inputs: (Batch, H, W, C) -> flat: (Batch*H*W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # RICERCA CODICI "MORTI" (Mode Collapse)
        # Un codice è "morto" se non viene quasi mai usato (cluster_size < threshold).
        # Se un codice non viene usato, è uno spreco di risorse perché l'encoder lo ignora.
        dead_indices = []
        
        # Iteriamo su tutto il codebook per controllare la frequenza d'uso di ogni vettore
        for i in range(self.num_embeddings):
            usage_count = self._ema_cluster_size[i]
            # Se l'uso è sotto la soglia (es. 1.0), lo segniamo come morto
            if usage_count < threshold:
                dead_indices.append(i)

        # Se non ci sono codici morti, usciamo subito
        if len(dead_indices) == 0:
            return 0
            
        # SELEZIONE NUOVI VETTORI
        # Se il codice X è inutile, buttiamolo via e sostituiamolo 
        # con un pixel reale preso a caso dall'immagine che stiamo vedendo ora.
        # Così diamo al codice una "seconda possibilità" di essere utile.
        num_dead = len(dead_indices)
        num_pixels = flat_input.size(0)
        
        # Non possiamo sostituire più codici di quanti pixel abbiamo a disposizione nel batch
        num_to_replace = min(num_dead, num_pixels)
        
        # Generiamo una lista di indici casuali per scegliere quali pixel copiare
        # randperm genera una permutazione casuale da 0 a num_pixels
        # genera una sequenza casuale di numeri interi da 0 a n-1. 
        # prende un intervallo di numeri e li "rimescola".
        random_pixel_indices = torch.randperm(num_pixels, device=inputs.device)[:num_to_replace]

        # SOSTITUZIONE (RESET)
        with torch.no_grad():
            # Iteriamo sui codici da sostituire uno per uno
            for i in range(num_to_replace):
                # L'indice del codebook che vogliamo resettare (es. il vettore 55 è morto)
                idx_to_reset = dead_indices[i]
                
                # L'indice del pixel casuale che diventerà il nuovo codice
                idx_random_pixel = random_pixel_indices[i]
                
                # Estraiamo il vettore del pixel (usiamo detach per non portarci dietro il gradiente)
                new_vector = flat_input[idx_random_pixel].detach()
                
                # Eseguiamo la sostituzione copiando i dati:
                
                # Il vettore principale del codebook diventa uguale a questo pixel
                self._embedding[idx_to_reset].copy_(new_vector)
                
                # Resettiamo anche il buffer ombra (usato per l'EMA)
                self._ema_w[idx_to_reset].copy_(new_vector)
                
                # Resettiamo il conteggio (cluster size) alla soglia minima
                # Questo evita che venga considerato di nuovo "morto" nel prossimo batch immediato
                # fill_() modifica i dati direttamente dove si trovano.
                self._ema_cluster_size[idx_to_reset].fill_(threshold)

        # Restituiamo il numero di codici che abbiamo riavviato (utile per i log)
        return len(dead_indices)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=64, commitment_cost=0.25, decay=0.99):
        super(VQVAE, self).__init__()
        
        # Salviamo i parametri principali
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # --- 1. ENCODER (Compressione) ---
        # L'obiettivo è ridurre l'immagine da 64x64 a 16x16 (un fattore 4x).
        # Usiamo Stride=2 per dimezzare le dimensioni spaziali ad ogni passaggio significativo.
        
        self.encoder = nn.Sequential(
            # === BLOCCO 1: Riduzione 64x64 -> 32x32 ===
            # Input: (Batch, 3 canali RGB, 64, 64)
            # Stride 2 riduce altezza e larghezza a metà
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64), # Normalizza i dati per stabilità
            nn.LeakyReLU(), # Funzione di attivazione che lascia passare piccoli valori negativi
            ResidualBlock(dim=64), # Aiuta a mantenere l'informazione in reti profonde

            # === BLOCCO 2: Riduzione 32x32 -> 16x16 ===
            # Input: (Batch, 64, 32, 32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(),
            ResidualBlock(dim=128),
            
            # === BLOCCO 3: Adattamento Canali (Resta 16x16) ===
            # Input: (Batch, 128, 16, 16)
            # Qui cambiamo solo il numero di canali (da 128 a embedding_dim)
            # Usiamo Stride 1, quindi le dimensioni spaziali non cambiano
            nn.Conv2d(in_channels=128, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=embedding_dim),
            nn.LeakyReLU(),
            ResidualBlock(dim=embedding_dim),
            
            # === BLOCCO FINALE: Raffinamento ===
            # Input: (Batch, embedding_dim=256, 16, 16)
            # Un'ultima convoluzione per preparare i dati al quantizzatore
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        # --- 2. VECTOR QUANTIZER (Il Dizionario) ---
        # Questo strato converte i vettori continui prodotti dall'encoder
        # in indici discreti del codebook.
        self.vq_layer = VectorQuantizerEMA(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost, 
            decay=decay
        )

        # --- 3. DECODER (Ricostruzione) ---
        # L'obiettivo è riportare l'immagine compressa (16x16) alla dimensione originale (64x64).
        # Usiamo ConvTranspose2d (convoluzione inversa) per l'upsampling (ingrandimento).
        
        self.decoder = nn.Sequential(
            # === BLOCCO INIZIALE: Elaborazione Post-Quantizzazione ===
            # Input: (Batch, embedding_dim, 16, 16)
            nn.Conv2d(in_channels=embedding_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(), 
            ResidualBlock(dim=128),
            
            # === BLOCCO 1: Elaborazione intermedia a 16x16 ===
            # Input: (Batch, 128, 16, 16)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(),
            ResidualBlock(dim=64),
            
            # === BLOCCO 2: Upsampling 16x16 -> 32x32 ===
            # Input: (Batch, 64, 16, 16)
            # ConvTranspose con Stride 2 raddoppia le dimensioni spaziali
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.LeakyReLU(),
            ResidualBlock(dim=32),
            
            # === BLOCCO 3: Upsampling 32x32 -> 64x64 ===
            # Input: (Batch, 32, 32, 32)
            # Ritorna a 3 canali (RGB)
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            
            # === ATTIVAZIONE FINALE ===
            # Usiamo Tanh per costringere i valori dei pixel nell'intervallo [-1, 1],
            # che è lo standard per la normalizzazione delle immagini.
            nn.Tanh() 
        )

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, vq_loss, perplexity, _ = self.vq_layer(z)
        x_recon = self.decoder(z_quantized)
        return x_recon, vq_loss, perplexity

    def get_codebook_indices(self, x):
        z = self.encoder(x)
        _, _, _, indices = self.vq_layer(z)
        return indices.view(x.shape[0], -1)
    
    def decode_from_indices(self, indices):
        z_q = F.embedding(indices, self.vq_layer._embedding)
        b, seq_len, dim = z_q.shape
        # Logica automatica: se seq_len=256 -> h=w=16. Se 64 -> h=w=8.
        h = w = int(seq_len**0.5) 
        z_q = z_q.permute(0, 2, 1).view(b, dim, h, w)
        return self.decoder(z_q)
    
    def restart_codebook(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            restarted_count = self.vq_layer.restart_unused_codes(z)
        return restarted_count