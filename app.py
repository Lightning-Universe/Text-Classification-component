#! pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component
#! mkdir -p ${HOME}/data/yelpreviewfull
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -o ${HOME}/data/yelpreviewfull/train.csv
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -o ${HOME}/data/yelpreviewfull/test.csv

import os
from copy import deepcopy

import lightning as L
from torch.optim import AdamW
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import (DriveTensorBoardLogger, MultiNodeLightningTrainerWithTensorboard,
                         TextClassificationDataLoader, TextDataset,
                         default_callbacks, get_default_clf_metrics,
                         warn_if_drive_not_empty, warn_if_local)


class TextClassification(L.LightningModule):
    def __init__(self, model, tokenizer, metrics=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if metrics is None:
            metrics = {}
        self.train_metrics = deepcopy(metrics)
        self.val_metrics = deepcopy(metrics)

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
        return AdamW(self.parameters(), lr=0.0001)


class MyTextClassification(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        warn_if_drive_not_empty(self.tensorboard_drive)
        warn_if_local()

        # --------------------
        # CONFIGURE YOUR MODEL
        # --------------------
        # Choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        # For local runs: Choose a small model (i.e. bloom-560m)
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
            dataset=TextDataset(
                csv_file=os.path.expanduser("~/data/yelpreviewfull/train.csv")
            ),
            tokenizer=tokenizer,
        )
        val_dataloader = TextClassificationDataLoader(
            dataset=TextDataset(
                csv_file=os.path.expanduser("~/data/yelpreviewfull/test.csv")
            ),
            tokenizer=tokenizer,
        )

        # -------------------
        # RUN YOUR FINETUNING
        # -------------------
        pl_module = TextClassification(
            model=module, tokenizer=tokenizer, metrics=get_default_clf_metrics(5)
        )

        # For local runs without multiple gpus, change strategy to "ddp"
        trainer = L.Trainer(
            max_epochs=2,
            limit_train_batches=100,
            limit_val_batches=100,
            strategy="deepspeed_stage_3_offload",
            precision=16,
            callbacks=default_callbacks(),
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
            log_every_n_steps=5,
        )
        trainer.fit(pl_module, train_dataloader, val_dataloader)


app = L.LightningApp(
    MultiNodeLightningTrainerWithTensorboard(MyTextClassification, 2, L.CloudCompute("gpu-fast-multi", disk_size=50))
)
