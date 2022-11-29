import os
from typing import Iterable, Tuple

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizer


class IterableTextClfDataset(IterableDataset):
    def __init__(self, dataset: Iterable[Tuple[int, str]]):
        self.dataset = dataset
        self.dataset_iter = None

    def __iter__(self):
        self.dataset_iter = iter(self.dataset)
        return self

    def __next__(self):
        data = next(self.dataset_iter)

        if isinstance(data, tuple) and isinstance(data[0], (int, float)):
            label, text, *_ = data
        else:
            label, text = None, data

        return dict(
            text=text,
            label=int(label - 1)
            if isinstance(label, (int, float))
            else None,  # labels need to start at 0
        )


class TextClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_token_len: int = 256,
        num_workers: int = min(os.cpu_count() - 1, 1),
    ):

        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers

    def prepare_data(self):
        # triggers potential downloads
        _ = next(iter(self.train_dataset))
        _ = next(iter(self.val_dataset))

        self.train_dataset = IterableTextClfDataset(self.train_dataset)
        self.val_dataset = IterableTextClfDataset(self.val_dataset)

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
