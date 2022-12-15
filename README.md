<div align="center">
    <h1>
        <img src="https://lightningaidev.wpengine.com/wp-content/uploads/2022/11/Asset-54-15.png">
        <br>
        Finetune large langugage models with Lightning
        </br>
    </h1>

<div align="center">

<p align="center">
  <a href="#run">Run</a> •
  <a href="https://www.lightning.ai/">Lightning AI</a> •
  <a href="https://lightning.ai/lightning-docs/">Docs</a>
</p>

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://lightning.ai/lightning-docs/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://www.pytorchlightning.ai/community)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)

</div>
</div>

______________________________________________________________________

Use Lightning Classify to pre-train or fine-tune a large language model for text classification, 
with as many parameters as you want (up to billions!). 

You can do this:
* using multiple GPUs
* across multiple machines
* on your own data
* all without any infrastructure hassle! 

All handled easily with the [Lightning Apps framework](https://lightning.ai/lightning-docs/).

## Run

To run paste the following code snippet in a file `app.py`:

```python
#! pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component
#! curl -L https://bit.ly/yelp_train --create-dirs -o ${HOME}/data/yelp/train.csv -C -
#! curl -L https://bit.ly/yelp_test --create-dirs -o ${HOME}/data/yelp/test.csv -C -

import lightning as L

import os, copy, torch

from transformers import BloomForSequenceClassification, BloomTokenizerFast
import lai_textclf as txtclf


class MyTextClassification(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        txtclf.warn_if_drive_not_empty(self.tensorboard_drive)
        txtclf.warn_if_local()

        # --------------------
        # CONFIGURE YOUR MODEL
        # --------------------
        # Choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        # For local runs: Choose a small model (i.e. bloom-560m)
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        num_labels = 5
        module = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        # -------------------
        # CONFIGURE YOUR DATA
        # -------------------
        data_path = os.path.expanduser("~/data/yelp")
        train_dataloader = txtclf.TextClassificationDataLoader(
            dataset=txtclf.TextDataset(csv_file=os.path.join(data_path, "train.csv")),
            tokenizer=tokenizer,
        )
        val_dataloader = txtclf.TextClassificationDataLoader(
            dataset=txtclf.TextDataset(csv_file=os.path.join(data_path, "test.csv")),
            tokenizer=tokenizer,
        )

        # -------------------
        # RUN YOUR FINETUNING
        # -------------------
        pl_module = TextClassification(model=module, tokenizer=tokenizer,
                                       metrics=txtclf.clf_metrics(num_labels))

        # For local runs without multiple gpus, change strategy to "ddp"
        trainer = L.Trainer(
            max_epochs=2, limit_train_batches=100, limit_val_batches=100,
            strategy="deepspeed_stage_3_offload", precision=16,
            callbacks=txtclf.default_callbacks(), log_every_n_steps=5,
            logger=txtclf.DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
        )
        trainer.fit(pl_module, train_dataloader, val_dataloader)


class TextClassification(L.LightningModule):
    def __init__(self, model, tokenizer, metrics=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_metrics = copy.deepcopy(metrics or {})
        self.val_metrics = copy.deepcopy(metrics or {})

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("train_loss", output.loss, prog_bar=True, on_epoch=True, on_step=True)
        self.train_metrics(output.logits, batch["labels"])
        self.log_dict(self.train_metrics, on_epoch=True, on_step=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("val_loss", output.loss, prog_bar=True)
        self.val_metrics(output.logits, batch["labels"])
        self.log_dict(self.val_metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)


component = txtclf.MultiNodeLightningTrainerWithTensorboard(
    MyTextClassification, num_nodes=2, cloud_compute=L.CloudCompute("gpu-fast-multi", disk_size=50)
)
app = L.LightningApp(component)
```

### Running on the cloud

```bash
lightning run app app.py --cloud
```

Don't want to use the public cloud? Contact us at `product@lightning.ai` for early access to run on your private cluster (BYOC)!


### Running locally (limited)
This example is optimized for the cloud. To run it locally on your laptop, choose a smaller model, and change the trainer settings like so:

```python
class MyTextClassification(L.LightningWork):
    def run(self):
        ...
        trainer = L.Trainer(strategy="ddp")
        ...
```
Then run the app with 

```bash
lightning run app app.py --setup
```

