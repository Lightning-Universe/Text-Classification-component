import csv
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        with open(csv_file, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            self.rows = list(reader)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        label, text = self.rows[item]
        label = int(label) - 1
        return dict(text=text, label=label)


class TextClassificationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_token_len: int = 256,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        collate_fn = TextEncodingCollate(tokenizer, max_token_len)
        num_workers = num_workers if num_workers is not None else os.cpu_count()
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )


class TextEncodingCollate:
    def __init__(self, tokenizer, max_sequence_length=256):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, samples):
        texts = [sample["text"] for sample in samples]
        text_encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
        )
        labels = torch.tensor([sample["label"] for sample in samples])
        return dict(
            **text_encodings,
            labels=labels,
        )
