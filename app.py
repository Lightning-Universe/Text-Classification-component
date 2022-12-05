#! pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component
#! mkdir -p ${HOME}/data/yelpreviewfull
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -o ${HOME}/data/yelpreviewfull/train.csv
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -o ${HOME}/data/yelpreviewfull/test.csv

import os

import lightning as L
from transformers import BloomForSequenceClassification, BloomTokenizerFast, AdamW

from utilities import default_callbacks, TextClassificationDataLoader, TextDataset


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

    def get_dataloaders(self, tokenizer):
        train_dataloader = TextClassificationDataLoader(
            dataset=TextDataset(csv_file=os.path.expanduser("~/data/yelpreviewfull/train.csv")),
            tokenizer=tokenizer,
        )
        val_dataloader = TextClassificationDataLoader(
            dataset=TextDataset(csv_file=os.path.expanduser("~/data/yelpreviewfull/test.csv")),
            tokenizer=tokenizer
        )
        return train_dataloader, val_dataloader

    def get_trainer(self):
        return L.Trainer(strategy="deepspeed_stage_3_offload", precision=16, callbacks=default_callbacks())

    def run(self):
        module, tokenizer = self.get_model()
        train_dataloader, val_dataloader = self.get_dataloaders(tokenizer)
        pl_module = TextClassification(model=module, tokenizer=tokenizer)
        trainer = self.get_trainer()
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
