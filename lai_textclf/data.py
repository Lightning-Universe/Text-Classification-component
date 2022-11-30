import csv

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class YelpReviewFull(Dataset):
    def __init__(self, csv_file: str):
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


class TextClassificationData(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_token_len: int = 256,
        num_workers: int = 2,
    ):

        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers

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
