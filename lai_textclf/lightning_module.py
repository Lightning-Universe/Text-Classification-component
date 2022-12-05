from typing import Sequence, Union

import lightning as L
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from torch.optim import AdamW


class TextClassification(LightningModule):
    """PyTorch Lightning Model class"""

    def __init__(
        self,
        model,
        tokenizer,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        loss, outputs = self.model(**batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self(**batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            verbose=True,
            mode="min",
        )
        checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        return [early_stopping, checkpoints]
