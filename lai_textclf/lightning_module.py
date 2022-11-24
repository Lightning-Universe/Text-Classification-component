# MIT License
#
# Copyright (c) 2021 Shivanand Roy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Code in this file is based on https://github.com/Shivanandroy/simpleT5 by Shivanand Roy
"""

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
        """forward step"""
        output = self.model(*args, **kwargs)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        """training step"""

        loss, outputs = self(**batch)

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )

        self.log(
            "max-gpu-mem-gb",
            torch.cuda.max_memory_allocated() // (1024**3),
            prog_bar=True,
            logger=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""

        loss, outputs = self(**batch)

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        """test step"""

        loss, outputs = self(**batch)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """configure optimizers"""
        return AdamW(self.parameters(), lr=0.0001)

    def predict_step(self, batch, batch_idx):
        return self(**batch)[1].argmax(-1) + 1


def predict(module: LightningModule, source_text: str):
    inputs = module.tokenizer(source_text, return_tensors="pt")
    with torch.no_grad():
        _, logits = module(**inputs)
    predicted_class_id = logits.argmax().item()
    return predicted_class_id
