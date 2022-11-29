# !pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component
import os
from pathlib import Path

import lightning as L
import torchtext.datasets
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import TextClf
from lai_textclf.data import TextClassificationDataModule
from lai_textclf.lightning_module import TextClassification


class MyTextClassification(L.LightningWork):
    def get_model(self, num_labels: int):
        # choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b

        # for huggingface/transformers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model_type = "bigscience/bloom-560m"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        return model, tokenizer

    def get_dataset(self):
        data_root_path = os.path.expanduser("~/.cache/torchtext/YelpReview")
        train_dset = torchtext.datasets.YelpReviewFull(
            root=os.path.join(data_root_path, "train"), split="train"
        )
        val_dset = torchtext.datasets.YelpReviewFull(
            root=os.path.join(data_root_path, "val"), split="test"
        )
        num_labels = 5
        return train_dset, val_dset, num_labels

    def get_trainer_settings(self):
        return dict(strategy="deepspeed_stage_3_offload", precision=16)

    def finetune(self):
        train_dset, val_dset, num_labels = self.get_dataset()
        module, tokenizer = self.get_model(num_labels)
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        datamodule = TextClassificationDataModule(
            train_dataset=train_dset, val_dataset=val_dset, tokenizer=tokenizer
        )
        trainer = L.Trainer(**self.get_trainer_settings())

        trainer.fit(pl_module, datamodule)

    def run(self):
        self.finetune()


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        MyTextClassification,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu", disk_size=50),
    )
)
