import csv
import os
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class TextClassificationData(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_token_len: int = 256,
        num_workers: Optional[int] = None,
    ):

        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TextEncodingCollate(self.tokenizer, self.max_token_len),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TextEncodingCollate(self.tokenizer, self.max_token_len),
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
