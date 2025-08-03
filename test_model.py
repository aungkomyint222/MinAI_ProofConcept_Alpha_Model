#!/usr/bin/env python3
"""
Test script to debug model loading issues
"""

import os
import sys
import traceback

def check_files():
    """Check if required files exist"""
    required_files = [
        "trained_model.pth",
        "burmese_tokenizer.model",
        "burmese_tokenizer.vocab"
    ]
    
    print("=== File Check ===")
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} (size: {size:,} bytes)")
        else:
            print(f"✗ {file_path} - NOT FOUND")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("=== Dependency Check ===")
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("sentencepiece", "SentencePiece"),
        ("gradio", "Gradio")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} - NOT INSTALLED: {e}")
    print()

def test_model_loading():
    """Test model loading"""
    print("=== Model Loading Test ===")
    try:
        from model import load_model
        print("Importing load_model function...")
        
        model = load_model()
        print("✓ Model loaded successfully!")
        
        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    print()

def test_tokenizer_loading():
    """Test tokenizer loading"""
    print("=== Tokenizer Loading Test ===")
    try:
        from model import load_tokenizer
        print("Importing load_tokenizer function...")
        
        tokenizer = load_tokenizer()
        print("✓ Tokenizer loaded successfully!")
        
        # Test encoding/decoding
        test_text = "Hello"
        encoded = tokenizer.encode(test_text, out_type=int)
        decoded = tokenizer.decode(encoded)
        print(f"Test encoding: '{test_text}' → {encoded} → '{decoded}'")
        
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    print()

def test_generation():
    """Test text generation"""
    print("=== Generation Test ===")
    try:
        import torch
        from model import load_model, load_tokenizer
        
        model = load_model()
        tokenizer = load_tokenizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test generation
        test_prompt = "Hello"
        input_ids = torch.tensor([tokenizer.encode(test_prompt, out_type=int)]).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.8
            )
        
        result = tokenizer.decode(output[0].tolist())
        print(f"✓ Generation test successful!")
        print(f"Input: '{test_prompt}'")
        print(f"Output: '{result}'")
        
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    print()

if __name__ == "__main__":
    print("MinAI Model Debug Test")
    print("=" * 50)
    
    check_files()
    check_dependencies()
    test_model_loading()
    test_tokenizer_loading()
    test_generation()
    
    print("Debug test completed!")
