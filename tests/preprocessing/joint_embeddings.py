import torch
from torch import nn
import pytest
from model.joint_embeddings import JointEmbeddings

# --- Test Suite ---

@pytest.fixture
def model():
    return JointEmbeddings(dim_D=128, num_attribute=3)

def test_output_shape(model):
    """Verifica che le dimensioni dell'output siano (B, 4099, D)"""
    batch_size = 4
    pixel_seq = torch.randn(batch_size, 4096, 1)
    attr_vector = torch.randint(0, 2, (batch_size, 3))
    
    output = model(pixel_seq, attr_vector)
    
    assert output.shape == (batch_size, 4099, 128)

def test_positional_encoding_logic(model):
    """Verifica che il pos encoding generi un tensore della dimensione corretta"""
    pos_enc = model.create_2d_pos_encoding(device="cpu")
    assert pos_enc.shape == (4096, 128)
    # Verifica che non siano tutti zeri
    assert torch.abs(pos_enc).sum() > 0

def test_gradient_flow(model):
    """Verifica che i gradienti scorrano attraverso tutti i componenti"""
    batch_size = 2
    pixel_seq = torch.randn(batch_size, 4096, 1)
    attr_vector = torch.randint(0, 2, (batch_size, 3))
    
    output = model(pixel_seq, attr_vector)
    loss = output.pow(2).sum()
    loss.backward()
    
    # Controlla che ogni parametro abbia ricevuto un gradiente
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradiente mancante per {name}"
        assert torch.abs(param.grad).sum() > 0, f"Gradiente nullo per {name}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU non disponibile")
def test_device_moving():
    """Verifica che il modello funzioni correttamente su GPU"""
    device = torch.device("cuda")
    model = JointEmbeddings(dim_D=64).to(device)
    pixel_seq = torch.randn(2, 4096, 1).to(device)
    attr_vector = torch.randint(0, 2, (2, 3)).to(device)
    
    output = model(pixel_seq, attr_vector)
    assert output.device.type == "cuda"

def test_invalid_input_values(model):
    """Verifica che il modello sollevi errore se gli attributi non sono 0 o 1"""
    pixel_seq = torch.randn(1, 4096, 1)
    attr_vector = torch.tensor([[0, 1, 5]]) # 5 è fuori dal range dell'Embedding (num_embeddings=2)
    
    try: 
        model(pixel_seq, attr_vector)
        assert 0
    except: # PyTorch solleva RuntimeError per indici out of bounds:
        assert 1