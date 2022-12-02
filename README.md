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
#! wget https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -O ${HOME}/data/yelpreviewfull/train.csv
#! wget https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -O ${HOME}/data/yelpreviewfull/test.csv
import os

import lightning as L
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import TextClassification, TextClassificationData, TextClf, YelpReviewFull


class MyTextClassification(TextClf):
    def get_model(self, num_labels: int):
        # choose from: bloom-560m (use to test locally), bloom-1b1, bloom-1b7, bloom-3b
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        return model, tokenizer

    def get_dataset(self):
        train_dset = YelpReviewFull(csv_file=os.path.expanduser("~/data/yelpreviewfull/train.csv"))
        val_dset = YelpReviewFull(csv_file=os.path.expanduser("~/data/yelpreviewfull/test.csv"))
        num_labels = 5
        return train_dset, val_dset, num_labels

    def get_trainer(self):
        # to test locally use L.Trainer(strategy="ddp", precision=16, accelerator="auto")
        return L.Trainer(strategy="deepspeed_stage_3_offload", precision=16)

    def finetune(self):
        train_dset, val_dset, num_labels = self.get_dataset()
        module, tokenizer = self.get_model(num_labels)
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        datamodule = TextClassificationData(
            train_dataset=train_dset, val_dataset=val_dset, tokenizer=tokenizer
        )
        trainer = self.get_trainer()

        trainer.fit(pl_module, datamodule)


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

