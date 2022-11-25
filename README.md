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

# ! pip install git+https://github.com/Lightning-AI/LAI-Text-Classification
import lightning as L
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import TextClf

sample_text = "Blue is the most beautiful color!"


class MyTextClassification(TextClf):
    def get_model(self):
        # choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=5, ignore_mismatched_sizes=True
        )
        return model, tokenizer

    def get_dataset_name(self) -> str:
        return "YelpReviewFull"

    def get_trainer_settings(self):
        settings = super().get_trainer_settings()
        settings["strategy"] = "deepspeed_stage_3_offload"
        settings["precision"] = "bf16"
        settings["max_epochs"] = 2
        settings["limit_train_batches"] = 10
        settings["limit_val_batches"] = 10
        return settings

    def run(self):
        super().run()
        if self.is_main_process:
            num_stars = self.predict(sample_text) + 1
            print("Review text:\n", sample_text)
            print("Predicted rating:", "★" * num_stars)


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        MyTextClassification,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast-multi", disk_size=50),
    )
)

```

### Running on cloud

```bash
lightning run app app.py --cloud
```

Don't want to use the public cloud? Contact us at `product@lightning.ai` for early access to run on your private cluster (BYOC)!


### Running locally (limited)

You can run also locally on a single node, but you will require multiple GPUs with enough memory depending on the
model size you choose.

```bash
lightning run app app.py --setup
```
