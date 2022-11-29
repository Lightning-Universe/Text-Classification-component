import os.path
from pathlib import Path

import torch
import torchtext.datasets
from torch.utils.data.datapipes.datapipe import IterDataPipe

from lai_textclf.data import (IterableTextClfDataset,
                              TextClassificationDataModule,
                              TextEncodingCollate)


def test_iterable_text_clf_dataset():
    data = [(1, "a"), (2, "b"), (3, "b")]

    dset = IterableTextClfDataset(dataset=data)
    counter = 0
    for sample in dset:

        assert isinstance(sample, dict)
        keys = list(sample.keys())
        assert len(keys) == 2
        assert "text" in keys
        assert "label" in keys

        assert sample["label"] == counter
        counter += 1

    assert counter == 3


class BoringTokenizer:
    def __call__(self, texts, max_length: int = 256, **kwargs):
        return {"input_ids": torch.rand(len(texts), max_length)}


def test_textclassification_datamodule():
    dm = TextClassificationDataModule(
        "IMDB", tokenizer=BoringTokenizer(), max_token_len=512
    )

    assert dm.dset_cls is None
    assert dm.train_dataset is None
    assert dm.val_dataset is None
    assert dm.test_dataset is None

    dm.prepare_data()
    assert dm.dset_cls is not None
    assert dm.dset_cls is torchtext.datasets.IMDB
    path = Path.home() / ".cache/torchtext/IMDB"
    assert os.path.isdir(path)

    dm.setup()

    assert isinstance(dm.train_dataset, IterableTextClfDataset)
    assert isinstance(dm.val_dataset, IterableTextClfDataset)
    assert isinstance(dm.test_dataset, IterableTextClfDataset)

    assert isinstance(dm.train_dataset.dataset, IterDataPipe)
    assert isinstance(dm.val_dataset.dataset, IterDataPipe)
    assert isinstance(dm.test_dataset.dataset, IterDataPipe)

    assert isinstance(dm.train_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(dm.val_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(dm.test_dataloader(), torch.utils.data.DataLoader)


def test_text_encoding_collate():
    collate = TextEncodingCollate(BoringTokenizer(), 512)

    output = collate(
        [
            {"text": "this is cool!", "label": 3},
            {"text": "this is super cool!", "label": 5},
        ]
    )
    assert isinstance(output, dict)
    assert isinstance(output["input_ids"], torch.Tensor)
    assert output["input_ids"].shape == (2, 512)
    assert isinstance(output["labels"], torch.Tensor)
    assert output["labels"].shape == (2,)
