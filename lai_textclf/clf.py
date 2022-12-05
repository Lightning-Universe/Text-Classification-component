from abc import ABC, abstractmethod
from typing import Any, Tuple

import lightning as L
import torch
import torch.nn as nn


class TextClf(L.LightningWork, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drive = L.app.storage.Drive("lit://artifacts")

    @abstractmethod
    def get_model(self) -> Tuple[nn.Module, Any]:
        """Return your large transformer language model here."""

    @abstractmethod
    def get_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Return the train and validation dataset for text classification and number of classification labels"""

    def get_trainer(self) -> L.Trainer:
        """Override this to change the Lightning Trainer default settings for finetuning."""
        return L.Trainer()

    @abstractmethod
    def finetune(self):
        """Carry out the finetuning process"""

    def run(self):
        self.finetune()
