"""
 !pip install 'git+https://github.com/Lightning-AI/LAI-TLDR'
"""  # fixme
import lightning as L
from transformers import BloomTokenizerFast, BloomForSequenceClassification

from lai_textclf.clf import TLDR

sample_text = """
ML Ops platforms come in many flavors from platforms that train models to platforms that label data and auto-retrain models. To build an ML Ops platform requires dozens of engineers, multiple years and 10+ million in funding. The majority of that work will go into infrastructure, multi-cloud, user management, consumption models, billing, and much more.
Build your platform with Lightning and launch in weeks not months. Focus on the workflow you want to enable (label data then train models), Lightning will handle all the infrastructure, billing, user management, and the other operational headaches.
"""


class GiveMeAName(TLDR):

    def get_model(self):
        # choices:
        # bloom-560m
        # bloom-1b1
        # bloom-1b7
        # bloom-3b
        # bloom-7b1
        model_type = "bigscience/bloom-560m"

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

    def run(self):
        super().run()

        # # Make a prediction at the end of fine-tuning
        # if self._trainer.global_rank == 0:
        #     predictions = predict(self._pl_module.to("cuda"), sample_text)
        #     print("Input text:\n", sample_text)
        #     print("Summarized text:\n", predictions[0])


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        GiveMeAName,
        num_nodes=1,  # Fixme
        cloud_compute=L.CloudCompute("gpu", disk_size=50),
    )
)
