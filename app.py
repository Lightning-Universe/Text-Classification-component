import lightning as L
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from lai_textclf import TextClf

sample_text = "Blue is the most beautiful color!"


class MyTextClassification(TextClf):

    def get_model(self):
        # choose from: bloom-560m, bloom-1b1, bloom-1b7, bloom-3b, bloom-7b1
        model_type = "bigscience/bloom-3b"
        tokenizer = BloomTokenizerFast.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = BloomForSequenceClassification.from_pretrained(model_type, num_labels=5, ignore_mismatched_sizes=True)

        # TODO: needed?
        # model.resize_token_embeddings(len(tokenizer))
        # model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer

    def get_dataset_name(self) -> str:
        return "YelpReviewFull"

    def get_trainer_settings(self):
        settings = super().get_trainer_settings()
        settings['strategy'] = "deepspeed_stage_3_offload"
        settings['precision'] = 'bf16'  # TODO: only supported on Ampere GPUs
        settings['max_steps'] = 10
        return settings

    def run(self):
        super().run()
        num_stars = self.predict(sample_text) + 1
        print("Review text:\n", sample_text)
        print("Predicted rating:", "★" * num_stars)


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        MyTextClassification,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast-multi", disk_size=50),
    )
)
