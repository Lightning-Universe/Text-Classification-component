
import lightning as L
import torch.cuda
from transformers import BloomTokenizerFast, BloomForSequenceClassification

from lai_textclf.clf import TextClf

sample_text = """
ML Ops platforms come in many flavors from platforms that train models to platforms that label data and auto-retrain models. To build an ML Ops platform requires dozens of engineers, multiple years and 10+ million in funding. The majority of that work will go into infrastructure, multi-cloud, user management, consumption models, billing, and much more.
Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want to enable (label data then train models), Lightning will handle all the infrastructure, billing, user management, and the other operational headaches.
"""


class GiveMeAName(TextClf):

    def get_model(self):
        # choices:
        # bloom-560m
        # bloom-1b1
        # bloom-1b7
        # bloom-3b
        # bloom-7b1
        model_type = "bigscience/bloom-3b"

        print(torch.cuda.get_device_name())

        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        num_labels = 5
        model = BloomForSequenceClassification.from_pretrained(model_type, num_labels=num_labels, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer

    def get_data_source(self) -> str:
        return "YelpReviewFull"

    def get_trainer_settings(self):
        settings = super().get_trainer_settings()

        from lightning.pytorch.strategies import DeepSpeedStrategy

        settings['strategy'] = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True, pin_memory=True)
        settings['precision'] = 'bf16'

        return settings

app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        GiveMeAName,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast-multi", disk_size=50),
    )
)
