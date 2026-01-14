import os
from PIL import Image
from tqdm import tqdm # Barra di avanzamento (pip install tqdm)
from config import *

# --- CONFIGURAZIONE ---
# Cartelle di input
INPUT_IMG_DIR = DATASET_CELEBA_IMGS
INPUT_ATTR_FILE = DATASET_CELEBA_LIST_ATTR

# Cartelle di output
OUTPUT_DIR = DATASET_CELEBA_PROCESSED_DATA
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, 'images_64x64')
OUTPUT_ATTR_FILE = os.path.join(OUTPUT_DIR, 'clean_attributes.txt')

# Parametri Immagine
TARGET_SIZE = (64, 64)
CROP_SIZE = (178, 178) # CelebA è 178x218. Ritagliamo un quadrato centrale.

# Attributi da mantenere
ATTR_TO_KEEP = ['Male', 'Smiling', 'Young']

def preprocess():
    # 1. Crea le cartelle di destinazione se non esistono
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
        print(f"Creata cartella: {OUTPUT_IMG_DIR}")

    print("Inizio lettura e filtro attributi...")
    
    # 2. Leggi il file degli attributi
    with open(INPUT_ATTR_FILE, 'r') as f:
        lines = f.readlines()


    headers = lines[1].split() # Nomi degli attributi
    
    # Trova gli indici delle colonne che ci interessano
    # Nota: headers non ha "image_id", quindi l'indice 0 corrisponde al primo attributo
    attr_indices = []
    for attr in ATTR_TO_KEEP:
        try:
            idx = headers.index(attr)
            attr_indices.append(idx)
        except ValueError:
            print(f"ERRORE: Attributo '{attr}' non trovato nel file!")
            return

    print(f"Indici trovati per {ATTR_TO_KEEP}: {attr_indices}")

    # 3. Prepara il nuovo file di testo
    processed_lines = []
    # Scriviamo l'header nel nuovo file
    header_line = "image_id " + " ".join(ATTR_TO_KEEP) + "\n"
    processed_lines.append(header_line)

    valid_images = 0

    # 4. Ciclo su tutte le righe (immagini)
    # Partiamo dalla riga 2 perché 0 è il count e 1 sono gli header
    print("Inizio elaborazione immagini...")
    
    for line in tqdm(lines[2:]):
        parts = line.split()
        image_name = parts[0]
        
        # Recupera i valori degli attributi selezionati
        # parts[0] è il nome file, quindi parts[idx + 1] è il valore dell'attributo
        current_attrs = [parts[i + 1] for i in attr_indices]
        
        # Conversione: Nel file originale -1 = No, 1 = Sì. 
        # Convertiamo in 0 = No, 1 = Sì per comodità con PyTorch
        converted_attrs = []
        for val in current_attrs:
            if val == '-1':
                converted_attrs.append('0')
            else:
                converted_attrs.append('1')
        
        # Percorso completo immagine sorgente
        src_path = os.path.join(INPUT_IMG_DIR, image_name)
        dst_path = os.path.join(OUTPUT_IMG_DIR, image_name)

        # Se l'immagine non esiste (può capitare se il download è incompleto), saltiamo
        if not os.path.exists(src_path):
            continue

        try:
            # --- ELABORAZIONE IMMAGINE ---
            with Image.open(src_path) as img:
                # 1. Converti in RGB (per sicurezza, alcune potrebbero essere scala di grigi)
                img = img.convert('RGB')
                
                # 2. Center Crop (178x178)
                # Calcolo coordinate: (left, top, right, bottom)
                width, height = img.size # 178, 218
                
                left = (width - CROP_SIZE[0]) / 2
                top = (height - CROP_SIZE[1]) / 2
                right = (width + CROP_SIZE[0]) / 2
                bottom = (height + CROP_SIZE[1]) / 2
                
                img = img.crop((left, top, right, bottom))
                
                # 3. Resize a 64x64
                img = img.resize(TARGET_SIZE, Image.BILINEAR)
                
                # 4. Salva
                img.save(dst_path, quality=95)
                
                # Aggiungi la riga al nuovo file di testo solo se l'immagine è stata salvata con successo
                new_line = f"{image_name} {' '.join(converted_attrs)}\n"
                processed_lines.append(new_line)
                valid_images += 1
                
        except Exception as e:
            print(f"Errore processando {image_name}: {e}")

    # 5. Scrivi il file degli attributi pulito
    with open(OUTPUT_ATTR_FILE, 'w') as f:
        # Aggiorniamo il conteggio (prima riga) con le immagini effettivamente processate
        f.write(f"{valid_images}\n")
        f.writelines(processed_lines)

    print("\n--- Finito! ---")
    print(f"Immagini salvate in: {OUTPUT_IMG_DIR}")
    print(f"Nuovo file attributi: {OUTPUT_ATTR_FILE}")
    print(f"Totale immagini processate: {valid_images}")

if __name__ == '__main__':
    preprocess()