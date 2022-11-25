import os
from abc import ABC, abstractmethod
from typing import Any, Tuple

import lightning as L
import torch
import torch.nn as nn

from lai_textclf.data import TextClassificationDataModule
from lai_textclf.lightning_module import TextClassification


class TextClf(L.LightningWork, ABC):
    """Finetune on a text summarization task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drive = L.app.storage.Drive("lit://artifacts")

    @property
    def is_main_process(self):
        return self._trainer is not None and self._trainer.global_rank == 0

    @abstractmethod
    def get_model(self) -> Tuple[nn.Module, Any]:
        """Return your large transformer language model here."""

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of a torchtext dataset for text classification."""

    def get_trainer_settings(self):
        """Override this to change the Lightning Trainer default settings for finetuning."""
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            verbose=True,
            mode="min",
        )
        checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            # dirpath="drive",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        return dict(
            callbacks=[early_stopping, checkpoints],
            strategy="deepspeed_stage_3",
            precision=16
        )

    def run(self):
        # for huggingface/transformers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        module, tokenizer = self.get_model()
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        datamodule = TextClassificationDataModule(
            dataset_name=self.get_dataset_name(), tokenizer=tokenizer
        )
        trainer = L.Trainer(**self.get_trainer_settings())

        self._pl_module = pl_module
        self._trainer = trainer

        trainer.fit(pl_module, datamodule)

        print("Uploading checkpoints and logs... It can take several minutes for very large models")
        for root, dirs, files in os.walk("lightning_logs", topdown=False):
            for name in files:
                self.drive.put(os.path.join(root, name))

    def predict(self, source_text):
        pl_module = self._pl_module
        if pl_module is None:
            module, tokenizer = self.get_model()
            pl_module = TextClassification(model=module, tokenizer=tokenizer)

        inputs = pl_module.tokenizer(source_text, return_tensors="pt")
        with torch.no_grad():
            _, logits = pl_module(**inputs)
        predicted_class_id = logits.argmax().item()
        return predicted_class_id
