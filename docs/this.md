1. Preprocessing (Spatial-Aware)
Input: Immagine CelebA $(B, 3, 178, 218)$
Task:
Center Crop: Ritaglio centrale a $(178, 178)$ per evitare distorsioni.
Resize: Ridimensionamento a $(64, 64)$.
Grayscale: Conversione a singolo canale (grigio).
Flattening: Trasformazione in vettore $(B, 4096, 1)$.
Output: Sequenza normalizzata $(0, 1)$ di dimensione $(B, 4096, 1)$.

2. Parallel Embedding & Encoding

Branch A: Conditional Prefix
Input: 3 Attributi binari.
Task: Passaggio attraverso nn.Embedding per mappare gli stati $(0, 1)$ in vettori densi.
Output: Prefix Tensor $(B, 3, D)$.

Branch B: Visual Projection & 2D Positioning
Input: Sequenza pixel $(B, 4096, 1)$.
Task:
Linear Projection: Proietta il singolo valore del pixel in dimensione $D$.
2D Learned Positional Encoding: Somma di due embedding (Row + Column) per iniettare la coordinata spaziale $(x, y)$ che il flattening aveva rimosso.
Output: Visual Tokens $(B, 4096, D)$.

3. Core Engine: Causal Mamba Stack

Input: Concatenazione dei due rami $\to$ Sequenza totale di $(B, 4099 - 1, D)$. ("- 1" perché dopo l'ultimo pixel non c'è nulla che il modello deve pedire)
Struttura: $N$ blocchi di Residual Mamba Layers (basati sul codice del Prof).
Logica: Causalità garantita. Ogni pixel viene generato basandosi sui 3 attributi e sui pixel precedenti.
Meccanismo: Utilizzo di pscan in training (veloce) e inference_step in generazione (auto-regressivo).
Output: Embeddings contestualizzati $(B, 4099, D)$.

4. Classifier (Predictive Head)

Input: Ultimi 4096 token della sequenza (esclude il prefisso).
Task:
Linear Mapping: Trasformazione da dimensione $D$ a $256$ (classi di intensità).
In Training: Cross-Entropy Loss su 256 classi.
In Inference: Scelta del valore tramite argmax o campionamento probabilistico.
Output: Tensor dei Logits $(B, 4096, 256)$.

5. Post-Processing

Input: Indici dei pixel predetti $(B, 4096)$.
Task:
Reshape: Ritorno alla griglia $(64, 64)$.
Visualizzazione: Conversione in immagine tramite OpenCV/PIL.
Output: Immagine generata $64 \times 64$ coerente con gli attributi di input.



