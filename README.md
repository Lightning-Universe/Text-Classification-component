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
#! mkdir -p ${HOME}/data/yelpreviewfull
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -o ${HOME}/data/yelpreviewfull/train.csv
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -o ${HOME}/data/yelpreviewfull/test.csv

import os

import lightning as L
from torch.optim import AdamW
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import default_callbacks, TextClassificationDataLoader, TextDataset


class TextClassification(L.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("train_loss", output.loss, prog_bar=True, on_epoch=True, on_step=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("val_loss", output.loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


class MyTextClassification(L.LightningWork):
    def run(self):
        # --------------------
        # CONFIGURE YOUR MODEL
        # --------------------
        # Choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        module = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=5, ignore_mismatched_sizes=True
        )

        # -------------------
        # CONFIGURE YOUR DATA
        # -------------------
        train_dataloader = TextClassificationDataLoader(
            dataset=TextDataset(csv_file=os.path.expanduser("~/data/yelpreviewfull/train.csv")),
            tokenizer=tokenizer,
        )
        val_dataloader = TextClassificationDataLoader(
            dataset=TextDataset(csv_file=os.path.expanduser("~/data/yelpreviewfull/test.csv")),
            tokenizer=tokenizer
        )

        # -------------------
        # RUN YOUR FINETUNING
        # -------------------
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        trainer = L.Trainer(
            max_steps=100, strategy="deepspeed_stage_3_offload", precision=16, callbacks=default_callbacks()
        )
        trainer.fit(pl_module, train_dataloader, val_dataloader)


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        MyTextClassification,
        num_nodes=2,
        cloud_compute=L.CloudCompute(
            name="gpu-fast-multi",
            disk_size=50,
        ),
    )
)

```

### Running on cloud

```bash
lightning run app app.py --cloud
```

Don't want to use the public cloud? Contact us at `product@lightning.ai` for early access to run on your private cluster (BYOC)!


### Running locally (limited)
This example is optimized for the cloud. To run it locally, choose a smaller model, change the trainer settings like so:

```python
class MyTextClassification(TextClf):
    ...
    
    def get_trainer(self):
        return dict(accelerator="cpu", devices=1)
```
This will avoid using the deepspeed strategy for training which is only compatible with multiple GPUs for model sharding.
Then run the app with 

```bash
lightning run app app.py --setup
```

