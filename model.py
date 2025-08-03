import torch
from transformers import GPT2Config, GPT2LMHeadModel
import sentencepiece as spm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    try:
        print("Creating model configuration...")
        config = GPT2Config(
            vocab_size=8000,
            n_positions=1024,
            n_ctx=1024,
            n_embd=256,
            n_layer=6,
            n_head=4
        )
        
        print("Initializing model...")
        model = GPT2LMHeadModel(config)
        
        # Check if model file exists
        model_path = "trained_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print(f"Moving model to device: {device}")
        model.to(device).eval()
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load tokenizer
def load_tokenizer():
    try:
        tokenizer_path = "burmese_tokenizer.model"
        
        # Check if tokenizer file exists
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found")
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        
        print("Tokenizer loaded successfully!")
        return sp
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
