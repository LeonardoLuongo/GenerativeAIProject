import torch
import sys

def main():
    print("--- DEBUG ALLINEAMENTO TOKEN (CPU ONLY) ---")
    
    # 1. SIMULAZIONE DATI (Uso numeri leggibili per capire cosa succede)
    # Immaginiamo un batch di size 1 per semplicità
    # Indici VQ-VAE: 256 token. Li chiameremo 100, 101, ... 355 per riconoscerli come "Pixel"
    # P0 = 100, P1 = 101, ... P255 = 355
    seq_len = 256
    indices = torch.arange(100, 100 + seq_len).unsqueeze(0) # (1, 256)
    
    # Attributi e SOS
    # Li chiamiamo 10, 20, 30 per gli attributi, 999 per SOS
    sos_token = torch.tensor([[[999]]]) # (1, 1, 1) -> simula embedding
    
    print(f"Input Indices (Pixel Reali): [ {indices[0,0]} ... {indices[0,-1]} ] (Len: {indices.shape[1]})")
    print("---------------------------------------------------------------")

    # 2. SIMULAZIONE FORWARD PASS (Come in mamba_vqvae.py)
    # Sequence Construction: [SOS, ATTR1, ATTR2, ATTR3, IMG_TOKENS...]
    # Lunghezza totale attesa: 1 (SOS) + 3 (Attr) + 256 (Img) = 260
    
    # Creiamo una lista fittizia di cosa "contiene" ogni step temporale
    # T0: SOS
    # T1: Attr1
    # T2: Attr2
    # T3: Attr3
    # T4: Pixel0 (Input 100)
    # ...
    # T259: Pixel255 (Input 355)
    
    full_seq_len = 1 + 3 + 256
    
    # Creiamo un tensore che rappresenta l'output del modello (Logits)
    # In Mamba/RNN, l'output allo step T predice l'input allo step T+1.
    # Quindi logits[T] contiene la predizione basata su Input[0...T]
    logits_simulation = torch.arange(0, full_seq_len).unsqueeze(0).unsqueeze(-1) # (1, 260, 1)
    
    print(f"Shape Logits Uscita Mamba: {logits_simulation.shape} (SOS+Attr+Img)")
    
    # 3. APPLICAZIONE SLICING DEL TUO TRAINING (train_mamba_vq.py)
    # Codice originale:
    # img_logits = logits[:, 4:-1, :]
    # target_seq = indices[:, 1:]
    
    print("\n--- ANALISI SLICING ATTUALE ---")
    print("Codice: img_logits = logits[:, 4:-1, :]")
    print("Codice: target_seq = indices[:, 1:]")
    
    slice_start = 4
    slice_end = -1 # Esclude l'ultimo
    
    # Simuliamo lo slicing
    sliced_logits = logits_simulation[:, slice_start:slice_end, :]
    sliced_targets = indices[:, 1:]
    
    print(f"\nShape Sliced Logits: {sliced_logits.shape}")
    print(f"Shape Sliced Targets: {sliced_targets.shape}")
    
    # 4. VERIFICA "CHI PREDICE COSA"
    # Ricordiamo: Logits all'indice N è l'output DOPO aver visto l'input all'indice N.
    # Deve predire l'input all'indice N+1.
    
    print("\n--- TABELLA DI VERITÀ ---")
    print(f"{'LOGIT INDEX':<12} | {'INPUT VISTO (Context)':<25} | {'PREDIZIONE (Target)':<20} | {'TARGET REALE':<20} | {'STATUS'}")
    print("-" * 105)
    
    # Mappatura indici nel vettore completo (0=SOS, 1=A1, 2=A2, 3=A3, 4=P0, 5=P1...)
    context_map = {
        0: "SOS", 1: "Attr1", 2: "Attr2", 3: "Attr3",
        4: "Pixel 0 (100)", 5: "Pixel 1 (101)", 258: "Pixel 254 (354)", 259: "Pixel 255 (355)"
    }
    
    # Analizziamo i primi 3 e gli ultimi 3 elementi del tuo slicing
    check_indices = [0, 1, 2, -3, -2, -1]
    
    for i in range(sliced_logits.shape[1]):
        # Indice reale nel tensore Logits originale
        original_idx = slice_start + i
        
        # Cosa ha visto il modello fino a qui?
        context_str = context_map.get(original_idx, f"Pixel {original_idx-4} ({100+original_idx-4})")
        
        # Qual è il target che gli stiamo dando?
        target_val = sliced_targets[0, i].item()
        target_str = f"Pixel ? ({target_val})"
        
        # Check correttezza
        # Se ho visto Pixel X, devo predire Pixel X+1.
        # Pixel 0 è 100. Pixel 1 è 101.
        prev_pixel_val = 100 + (original_idx - 4)
        expected_target = prev_pixel_val + 1
        
        is_correct = (target_val == expected_target)
        status = "✅ OK" if is_correct else "❌ WRONG"
        
        if i < 3 or i >= sliced_logits.shape[1] - 3:
            print(f"{original_idx:<12} | {context_str:<25} | Predicts -> {expected_target:<10} | {target_val:<20} | {status}")

    print("-" * 105)

    # 5. CONCLUSIONE
    print("\n--- DIAGNOSI ---")
    print("1. L'allineamento P1 -> P1 è CORRETTO.")
    print("   (Il modello vede P0 e predice P1).")
    print("2. ⚠️ IL PRIMO PIXEL (P0) VIENE SALTATO!")
    print("   Stai iniziando lo slice a '4' (Input visto: P0).")
    print("   L'indice '3' (Input visto: Attr3) è quello che dovrebbe predire P0.")
    print("   Attualmente il modello NON impara a generare il primo pixel dagli attributi.")
    print("   Il primo pixel che impara è P1, basandosi su P0 (teacher forcing).")
    
    print("\n--- CORREZIONE SUGGERITA ---")
    print("Cambia in train_mamba_vq.py:")
    print("DA:")
    print("   img_logits = logits[:, 4:-1, :]")
    print("   target_seq = indices[:, 1:]")
    print("A:")
    print("   img_logits = logits[:, 3:-1, :]  <-- Inizia da 3 (Dopo Attr3)")
    print("   target_seq = indices             <-- Target completo (include P0)")

if __name__ == "__main__":
    main()