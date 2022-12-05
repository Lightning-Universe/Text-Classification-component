import lightning as L
from torch.optim import AdamW


class TextClassification(L.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def finetune_step(self, model, batch):
        # Meant to get replaced by the real method provided by the Work
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.finetune_step(model=self.model, batch=batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.finetune_step(model=self.model, batch=batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
