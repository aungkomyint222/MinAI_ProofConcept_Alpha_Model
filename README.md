# MinAI Proof of Concept Alpha Model 🇲🇲

A lightweight GPT-2 based language model trained for Burmese text generation, serving as a proof of concept for minimal AI architectures.

## 🚀 Overview

This project demonstrates a compact yet functional language model specifically designed for the Burmese language. Built on the GPT-2 architecture, this alpha model showcases how transformer-based models can be scaled down while maintaining reasonable performance for text generation tasks.

## 📊 Model Specifications

### Architecture Details
- **Base Architecture**: GPT-2 Language Model
- **Total Parameters**: 7,049,216 (7M parameters)
- **Model Type**: Causal Language Model (CLM)
- **Framework**: PyTorch + Transformers

### Key Configurations
| Parameter | Value | Description |
|-----------|--------|-------------|
| **Vocabulary Size** | 8,000 | Custom Burmese tokenizer vocabulary |
| **Embedding Dimension** | 256 | Hidden state size |
| **Context Length** | 1,024 | Maximum sequence length |
| **Transformer Layers** | 6 | Number of decoder blocks |
| **Attention Heads** | 4 | Multi-head attention |
| **Feed-Forward Dimension** | 1,024 | MLP inner dimension |

### Model Components
```
GPT2LMHeadModel(
  ├── transformer: GPT2Model
  │   ├── wte: Token Embedding (8000 × 256)
  │   ├── wpe: Position Embedding (1024 × 256)
  │   ├── drop: Dropout (p=0.1)
  │   ├── h: 6 × GPT2Block
  │   │   ├── ln_1: LayerNorm
  │   │   ├── attn: Multi-Head Attention
  │   │   ├── ln_2: LayerNorm
  │   │   └── mlp: Feed-Forward Network
  │   └── ln_f: Final LayerNorm
  └── lm_head: Linear (256 → 8000)
)
```

## 🛠️ Technical Implementation

### Tokenization
- **Tokenizer**: SentencePiece
- **Model File**: `burmese_tokenizer.model`
- **Vocabulary**: Custom trained on Burmese text corpus
- **Special Tokens**: BOS (50256), EOS (50256), PAD (0)

### Model Configuration
```json
{
  "activation_function": "gelu_new",
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "resid_pdrop": 0.1,
  "layer_norm_epsilon": 1e-05,
  "initializer_range": 0.02,
  "transformers_version": "4.54.0"
}
```

## 🎯 Features

- **Lightweight Architecture**: Only 7M parameters for efficient inference
- **Burmese Language Support**: Custom tokenizer optimized for Myanmar script
- **Web Interface**: Gradio-based chatbot for easy interaction
- **GPU/CPU Compatible**: Automatic device detection and optimization
- **Text Generation**: Configurable sampling with temperature, top-k, and top-p

## � Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dependencies
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers library
- `sentencepiece` - Tokenization library
- `gradio` - Web interface framework

### Running the Model

1. **Load and Test the Model**:
```python
from model import load_model, load_tokenizer

model = load_model()
tokenizer = load_tokenizer()
```

2. **Launch Web Interface**:
```bash
python app.py
```

3. **Generate Text Programmatically**:
```python
import torch

# Encode input
input_text = "Your Burmese prompt here"
input_ids = torch.tensor([tokenizer.encode(input_text, out_type=int)])

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

# Decode output
generated_text = tokenizer.decode(output[0].tolist())
```

##  Project Structure

```
MinAI_ProofConcept_Alpha_Model/
├── README.md                 # This file
├── app.py                   # Gradio web interface
├── model.py                 # Model loading utilities
├── requirements.txt         # Python dependencies
├── trained_model.pth        # Trained model weights
├── burmese_tokenizer.model  # SentencePiece tokenizer
├── burmese_tokenizer.vocab  # Tokenizer vocabulary
└── about.txt               # Detailed model specifications
```

## � Usage Examples

### Web Interface
The Gradio interface provides an intuitive chatbot experience:
- Navigate to the web interface after running `python app.py`
- Type Burmese text in the input box
- Receive AI-generated responses
- Clear conversation history as needed

### Generation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 50 | Maximum tokens to generate |
| `temperature` | 0.8 | Sampling randomness (0.1-2.0) |
| `top_k` | 50 | Top-k sampling |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `do_sample` | True | Enable sampling vs greedy |

## 🔬 Model Performance

### Computational Requirements
- **Memory Usage**: ~27MB model file
- **Inference Speed**: Fast on modern CPUs/GPUs
- **Training Efficiency**: Suitable for resource-constrained environments

### Limitations
- **Context Window**: Limited to 1,024 tokens
- **Language Scope**: Primarily trained on Burmese text
- **Model Size**: Smaller vocabulary may limit expressiveness
- **Training Data**: Performance depends on training corpus quality

## 🎯 Use Cases

This proof of concept model is suitable for:
- **Research**: Studying minimal transformer architectures
- **Prototyping**: Quick Burmese NLP application development
- **Education**: Learning transformer model implementation
- **Resource-Constrained Deployment**: Edge computing scenarios
- **Language Preservation**: Supporting low-resource languages

## 🔧 Customization

### Adjusting Generation
Modify generation parameters in `app.py`:
```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,      # Increase for longer outputs
    temperature=0.7,         # Lower for more focused text
    top_k=40,               # Adjust sampling diversity
    top_p=0.9               # Nucleus sampling threshold
)
```

### Model Configuration
Update model architecture in `model.py`:
```python
config = GPT2Config(
    vocab_size=8000,        # Vocabulary size
    n_embd=256,            # Embedding dimension
    n_layer=6,             # Number of layers
    n_head=4               # Attention heads
)
```

## 🤝 Contributing

This is a proof of concept project. Potential improvements:
- Expand training dataset
- Increase model size for better performance
- Add multilingual support
- Implement fine-tuning capabilities
- Optimize inference speed

## ⚖️ License

This project is intended for research and educational purposes. Please ensure compliance with applicable licenses for dependencies and training data.

## � Citation

If you use this model in your research, please cite:
```
MinAI Proof of Concept Alpha Model - Burmese GPT-2
A lightweight transformer model for Myanmar language generation
```

---

**Note**: This is an alpha version proof of concept model. Performance may vary depending on input complexity and use case requirements.
