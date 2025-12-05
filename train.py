import os
import math
import time
import argparse
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from model import LlamaConfig, LlamaForCausalLM
from dataset import TextDataset, TextDatasetFromFile, create_dataloader


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    max_steps: int = 1000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    global_step: int = 0,
) -> tuple:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        lr = get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        _, loss = model(input_ids, labels=labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        progress_bar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{lr:.2e}",
        })
    
    return total_loss / num_batches, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        _, loss = model(input_ids, labels=labels)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    config: LlamaConfig,
    path: str,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "config": config.__dict__,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint["loss"]


def generate_sample(
    model: nn.Module,
    prompt: str,
    device: torch.device,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens, temperature, top_k)
    return tokenizer.decode(output_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Train a small Llama-style language model")
    parser.add_argument("--data_file", type=str, default="data/input.txt", help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length / block size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Model hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not os.path.exists(args.data_file):
        print(f"Data file not found: {args.data_file}")
        print("Creating sample dataset...")
        os.makedirs(os.path.dirname(args.data_file) if os.path.dirname(args.data_file) else "data", exist_ok=True)
        sample_text = """The quick brown fox jumps over the lazy dog. 
This is a sample text for training a small language model.
Machine learning is transforming the way we interact with technology.
Neural networks can learn complex patterns from data.
Language models have become increasingly powerful in recent years.
Training these models requires significant computational resources.
However, smaller models can still learn interesting patterns.
The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow models to focus on relevant parts of the input.
Self-attention is a key component of transformer models.
"""
        with open(args.data_file, "w", encoding="utf-8") as f:
            f.write(sample_text * 100)
        print(f"Sample dataset created at {args.data_file}")

    print("Loading dataset...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(len(text) * (1 - args.val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = TextDataset(train_text, block_size=args.block_size)
    val_dataset = TextDataset(val_text, block_size=args.block_size)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    config = LlamaConfig(
        vocab_size=train_dataset.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,
        max_position_embeddings=args.block_size,
        dropout=args.dropout,
    )

    print("\nModel Configuration:")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Block size: {config.max_position_embeddings}")
    print(f"  Dropout: {config.dropout}")

    model = LlamaForCausalLM(config).to(device)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, global_step, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    max_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps

    print(f"\nTraining for {args.epochs} epochs ({max_steps:,} steps)")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print("-" * 50)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        train_loss, global_step = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=args.warmup_steps,
            max_steps=max_steps,
            max_lr=args.learning_rate,
            min_lr=args.min_lr,
            global_step=global_step,
        )
        
        val_loss = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {math.exp(val_loss):.2f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, global_step, val_loss, config,
                os.path.join(args.checkpoint_dir, "best_model.pt")
            )
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, global_step, val_loss, config,
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            )
        
        prompt = "The"
        sample = generate_sample(model, prompt, device, train_dataset.tokenizer, max_new_tokens=50)
        print(f"\nSample generation (prompt='{prompt}'):")
        print(f"  {sample[:200]}...")
        print("-" * 50)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")


if __name__ == "__main__":
    main()
