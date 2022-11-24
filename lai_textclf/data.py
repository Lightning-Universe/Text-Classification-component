from typing import Iterable, Tuple
from pathlib import Path
import torch
import os
import torchtext
import inspect
from lightning.pytorch import LightningDataModule

from torch.utils.data import DataLoader, IterableDataset
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
    """PyTorch Lightning data class"""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_token_len: int = 256,
        num_workers: int = min(os.cpu_count() - 1, 1),
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            dataset_name: the torchtext dataset name to use
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            max_token_len (int, optional): max token length of source text. Defaults to 512.

        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.dset_cls = None
        self.train_split = None
        self.val_split = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup_data(self, download: bool = False):
        self.dset_cls = getattr(torchtext.datasets, self.dataset_name)

        choices_split = inspect.signature(self.dset_cls).parameters["split"].default

        self.train_split = choices_split[0]
        self.val_split = choices_split[1]
        self.test_split = choices_split[-1]

        data_root_dir = Path.home() / f".cache/torchtext/{self.dataset_name}"
        train_dset = self.dset_cls(root=data_root_dir, split=self.train_split)
        val_dset = self.dset_cls(root=data_root_dir, split=self.val_split)
        test_dset = self.dset_cls(root=data_root_dir, split=self.val_split)

        if download:
            print("Downloading Data, this may take some time. Please be patient!")

            # triggers downloads
            _ = next(iter(train_dset))
            _ = next(iter(val_dset))
            _ = next(iter(test_dset))

        self.train_dataset = IterableTextClfDataset(train_dset)
        self.val_dataset = IterableTextClfDataset(val_dset)
        self.test_dataset = IterableTextClfDataset(test_dset)

    def prepare_data(self):
        self.setup_data(download=True)

    def setup(self, stage=None):
        self.setup_data(download=False)

    def train_dataloader(self):
        """training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TextEncodingCollate(self.tokenizer, self.max_token_len),
        )

    def test_dataloader(self):
        """test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TextEncodingCollate(self.tokenizer, self.max_token_len),
        )

    def val_dataloader(self):
        """validation dataloader"""
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
