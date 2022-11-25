# !pip install git+https://github.com/Lightning-AI/LAI-Text-Classification-Component

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
        cloud_compute=L.CloudCompute(
            name="gpu-fast-multi",
            disk_size=50,
            mounts=L.storage.Mount(
                source="s3://pl-flash-data/lai-llm/lai-text-classification/weights/",
                mount_path="/home/zeus/.cache/huggingface/hub"
            )
            # Or we could set environ var TRANSFORMERS_CACHE = '/mnt/...' to avoid explicit home prefix
        ),
    )
)
