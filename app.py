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
        train_dset = YelpReviewFull(csv_file=os.path.expanduser("~/data/yelpreviewfull/train.csv"))
        val_dset = YelpReviewFull(csv_file=os.path.expanduser("~/data/yelpreviewfull/test.csv"))
        num_labels = 5
        return train_dset, val_dset, num_labels

    def get_trainer(self):
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
