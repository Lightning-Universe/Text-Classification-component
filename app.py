
import os

import lightning as L
import torchtext.datasets
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import TextClassification, TextClassificationData, TextClf
from torchtext.data.functional import to_map_style_dataset



class MyTextClassification(TextClf):
    def get_model(self, num_labels: int):
        # choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        return model, tokenizer

    def get_dataset(self):
        data_root_path = os.path.expanduser("~/.cache/torchtext")
        train_dset = torchtext.datasets.YelpReviewFull(
            root=data_root_path, split="train"
        )
        val_dset = torchtext.datasets.YelpReviewFull(root=data_root_path, split="test")
        num_labels = 5
        return to_map_style_dataset(train_dset), to_map_style_dataset(val_dset), num_labels

    def get_trainer_settings(self):
        return dict(strategy="deepspeed_stage_3_offload", precision=16)

    def finetune(self):
        train_dset, val_dset, num_labels = self.get_dataset()
        module, tokenizer = self.get_model(num_labels)
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        datamodule = TextClassificationData(
            train_dataset=train_dset, val_dataset=val_dset, tokenizer=tokenizer
        )
        trainer = L.Trainer(**self.get_trainer_settings())

        trainer.fit(pl_module, datamodule)


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        MyTextClassification,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu", disk_size=50),
    )
)
