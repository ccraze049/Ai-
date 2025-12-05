import os
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import Optional, List, Tuple


class TextDataset(Dataset):
    def __init__(
        self,
        text: str,
        block_size: int = 128,
        tokenizer_name: str = "gpt2",
    ):
        self.block_size = block_size
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        self.tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        
        print(f"Dataset: {len(self.tokens):,} tokens")
        print(f"Vocab size: {self.tokenizer.n_vocab:,}")

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab


class TextDatasetFromFile(TextDataset):
    def __init__(
        self,
        file_path: str,
        block_size: int = 128,
        tokenizer_name: str = "gpt2",
    ):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        super().__init__(text, block_size, tokenizer_name)


def create_dataloader(
    dataset: TextDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def split_dataset(
    text: str,
    train_ratio: float = 0.9,
    block_size: int = 128,
    tokenizer_name: str = "gpt2",
) -> Tuple[TextDataset, TextDataset]:
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_dataset = TextDataset(train_text, block_size, tokenizer_name)
    val_dataset = TextDataset(val_text, block_size, tokenizer_name)
    
    return train_dataset, val_dataset
