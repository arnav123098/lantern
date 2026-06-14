from lantern.tokenizer import Tokenizer
from pathlib import Path
import torch
from lantern.data.prep.dataset_manager import datasets

def split_text(filepath: str | Path, tokenizer: Tokenizer, val_split_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    with open(filepath, 'r') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens)

    n_train = int(tokens.size(-1) * (1 - val_split_size))
    train, val = tokens[:n_train], tokens[n_train:]

    return train, val

def split_shards(dataset_path: str | Path, tokenizer: Tokenizer, val_split_size: float) -> tuple[list[str], list[str]]:
    shards = datasets.get_files(dataset_path)

    n_train = int(len(shards) * (1 - val_split_size))
    train, val = shards[:n_train], shards[n_train:]

    return train, val
