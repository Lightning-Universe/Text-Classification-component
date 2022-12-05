#! pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component
#! mkdir -p ${HOME}/data/yelpreviewfull
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -o ${HOME}/data/yelpreviewfull/train.csv
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -o ${HOME}/data/yelpreviewfull/test.csv

import csv
import os

import lightning as L
from torch.utils.data import Dataset
from transformers import BloomForSequenceClassification, BloomTokenizerFast, AdamW

from utilities import TextClassificationDataModule, default_callbacks


class TextDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        with open(csv_file, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            self.rows = list(reader)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        label, text = self.rows[item]
        label = int(label) - 1
        return dict(text=text, label=label)


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
    num_classes = 5

    def get_model(self):
        # choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = BloomForSequenceClassification.from_pretrained(
            model_type, num_labels=self.num_classes, ignore_mismatched_sizes=True
        )
        return model, tokenizer

    def get_dataset(self):
        train_dset = TextDataset(csv_file=os.path.expanduser("~/data/yelpreviewfull/train.csv"))
        val_dset = TextDataset(csv_file=os.path.expanduser("~/data/yelpreviewfull/test.csv"))
        return train_dset, val_dset

    def get_trainer(self):
        return L.Trainer(strategy="deepspeed_stage_3_offload", precision=16, callbacks=default_callbacks())

    def finetune(self):
        train_dset, val_dset = self.get_dataset()
        module, tokenizer = self.get_model()
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        datamodule = TextClassificationDataModule(
            train_dataset=train_dset, val_dataset=val_dset, tokenizer=tokenizer
        )
        trainer = self.get_trainer()
        trainer.fit(pl_module, datamodule)

    def run(self):
        self.finetune()


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
