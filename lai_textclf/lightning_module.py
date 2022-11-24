import torch
from lightning.pytorch import LightningModule
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

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        loss, outputs = self(**batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log(
            "max-gpu-mem-gb",
            torch.cuda.max_memory_allocated() // (1024**3),
            prog_bar=True,
            logger=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self(**batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, outputs = self(**batch)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """configure optimizers"""
        return AdamW(self.parameters(), lr=0.0001)

    def predict_step(self, batch, batch_idx):
        return self(**batch)[1].argmax(-1) + 1
