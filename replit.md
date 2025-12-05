# Llama-Style Language Model Training

## Overview
A PyTorch training script for a small Llama-style transformer language model. This project implements a complete training pipeline including model architecture, dataset handling, and training loop with all standard optimizations.

## Project Structure
```
├── model.py          # Llama model architecture (RoPE, RMSNorm, SwiGLU)
├── dataset.py        # Text dataset and dataloader utilities
├── train.py          # Training script with full training loop
├── data/
│   └── input.txt     # Sample training text data
└── checkpoints/      # Saved model checkpoints (created during training)
```

## Model Architecture
The model implements a Llama-style transformer with:
- **RMSNorm**: Root Mean Square Layer Normalization
- **Rotary Position Embeddings (RoPE)**: Position encoding in attention
- **SwiGLU Activation**: Gated linear unit with SiLU activation
- **Grouped Query Attention**: Support for multi-query attention patterns
- **Weight Tying**: Shared embeddings between input and output layers

## Usage

### Basic Training
```bash
python train.py --data_file data/input.txt --epochs 10
```

### Custom Configuration
```bash
python train.py \
  --data_file data/your_text.txt \
  --epochs 20 \
  --batch_size 32 \
  --hidden_size 256 \
  --num_layers 6 \
  --num_heads 8 \
  --learning_rate 3e-4 \
  --block_size 128
```

### Key Parameters
- `--data_file`: Path to training text file
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--block_size`: Context length (sequence length)
- `--hidden_size`: Model hidden dimension
- `--num_layers`: Number of transformer layers
- `--num_heads`: Number of attention heads
- `--learning_rate`: Maximum learning rate
- `--dropout`: Dropout rate for regularization
- `--checkpoint_dir`: Directory to save checkpoints

## Features
- Cosine learning rate schedule with warmup
- Gradient clipping
- Gradient accumulation for effective larger batch sizes
- Automatic checkpoint saving
- Validation loss tracking with perplexity
- Sample text generation during training

## Dependencies
- PyTorch
- NumPy
- tqdm (progress bars)
- tiktoken (GPT-2 tokenizer)
