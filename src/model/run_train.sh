#!/bin/bash
#SBATCH --partition=gpuq             # Richiediamo la partizione con GPU NVIDIA [cite: 171]
#SBATCH -A did_generative_ai_336     # Il tuo account per il corso [cite: 245]
#SBATCH --job-name=train_diffusion
#SBATCH --output=train_log_%j.out    # File di log (stdout). %j sarà l'ID del job
#SBATCH --error=train_err_%j.err     # File di errori (stderr)
#SBATCH --ntasks=1                   # 1 task principale [cite: 246]
#SBATCH --cpus-per-task=4            # 4 CPU per velocizzare il caricamento dati [cite: 247]
#SBATCH --gpus-per-task=1            # 1 GPU dedicata [cite: 248]
#SBATCH --time=07:00:00              # Tempo massimo (es. 8 ore) [cite: 317]
#SBATCH --exclude=gnode07

# 1. Impostiamo l'interprete Python fornito dal prof (ambiente PyTorch/CUDA)
export PYTHON=/home/pfoggia/GenerativeAI/torch_cuda.env/bin/python
export TORCH_DEVICE=cuda

# 2. Ci spostiamo nella cartella corretta dove risiede il codice
# (Fondamentale perché Train.py deve importare Network.py e config.py che sono lì)
cd /home/F.MONDA2/Gen_AI_Project/GenerativeAIProject/src/model

# 3. Stampiamo info di debug e lanciamo il training
echo "Inizio training su nodo: $(hostname)"
echo "Data: $(date)"
echo "--- INFO NODO ---"
hostname
echo "--- INFO GPU ---"
nvidia-smi  # <--- FONDAMENTALE: Ci dice se la GPU è piena o libera
echo "-----------------"
$PYTHON Train.py

echo "Training terminato."