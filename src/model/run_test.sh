#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH -A did_generative_ai_336
#SBATCH --job-name=test_diffusion
#SBATCH --output=test_log_%j.out
#SBATCH --error=test_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:04:30           # 30 secondi sono più che sufficienti
#SBATCH --exclude=gnode07         # Evitiamo sempre il nodo rotto

# Setup Ambiente
export PYTHON=/home/pfoggia/GenerativeAI/torch_cuda.env/bin/python
export TORCH_DEVICE=cuda

# Vai alla cartella
cd /home/F.MONDA2/Gen_AI_Project/GenerativeAIProject/src/model

echo "--- START TEST ---"
echo "Nodo: $(hostname)"

# Esegui lo script di test
$PYTHON Test_Attribute.py

echo "--- END TEST ---"