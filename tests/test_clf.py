from collections import namedtuple
from typing import Tuple

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lai_textclf.clf import TextClf

return_type = namedtuple("return_type", ("loss", "logits"))


class BoringModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.layer = torch.nn.Linear(1, num_labels)

    def forward(self, *args, **kwargs):
        logits = self.layer(torch.rand(1))
        loss = logits.sum()
        return return_type(loss, logits)


class BoringTokenizer:
    def __call__(self, texts, max_length: int = 256, **kwargs):
        return {"input_ids": torch.rand(len(texts), max_length)}


class MyTextClf(TextClf):
    def __init__(self):
        super().__init__()
        self.drive.component_name = "DummyComponent"

    def get_model(self, num_labels):
        return BoringModel(num_labels), BoringTokenizer()

    def get_trainer_settings(self):
        settings = super().get_trainer_settings()
        settings.pop("strategy")
        settings["max_epochs"] = 1
        settings["limit_train_batches"] = 2
        settings["limit_val_batches"] = 2
        return settings

    def get_dataset_name(self) -> Tuple[str, int]:
        return "IMDB", 2


def test_class_instantiation():
    MyTextClf()


def test_trainer_settings():
    settings = MyTextClf().get_trainer_settings()

    assert isinstance(settings["callbacks"][0], EarlyStopping)
    assert isinstance(settings["callbacks"][1], ModelCheckpoint)
    assert len(settings["callbacks"]) == 2

    assert settings["precision"] == 16
    assert settings["max_epochs"] == 1
    assert settings["limit_train_batches"] == 2
    assert settings["limit_val_batches"] == 2

    Trainer(**settings)


def test_training():
    clf = MyTextClf()
    clf.run()
    clf.predict("This is a great Test!")
