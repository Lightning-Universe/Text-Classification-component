#! pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component
#! mkdir -p ${HOME}/data/yelpreviewfull
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -o ${HOME}/data/yelpreviewfull/train.csv
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -o ${HOME}/data/yelpreviewfull/test.csv

import os

import lightning as L
from lightning.app.storage import Drive
from torch.optim import AdamW
from transformers import BloomForSequenceClassification, BloomTokenizerFast

from lai_textclf import default_callbacks, TextClassificationDataLoader, TextDataset, get_logger, TensorBoardWrapperFlow


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

    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        # --------------------
        # CONFIGURE YOUR MODEL
        # --------------------
        # Choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b
        model_type = "bigscience/bloom-560m"
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
            max_steps=10000, strategy="deepspeed_stage_3_offload", precision=16, callbacks=default_callbacks(),
            logger=get_logger(save_dir=".", drive=self.tensorboard_drive)
        )
        trainer.fit(pl_module, train_dataloader, val_dataloader)


tb_drive = Drive("lit://tb_drive")
app = L.LightningApp(
    TensorBoardWrapperFlow(
        tb_drive,
        L.app.components.LightningTrainerMultiNode(
            MyTextClassification,
            num_nodes=1,
            cloud_compute=L.CloudCompute(
                name="gpu-fast-multi",
                disk_size=50,
            ),
            tb_drive=tb_drive
        )
    )
)
