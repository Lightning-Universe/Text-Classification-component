from lightning import Trainer

from lai_textclf.data import TextClassificationDataModule
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from lai_textclf.lightning_module import TextClassification

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
dm = TextClassificationDataModule('YelpReviewFull', tokenizer)
num_labels = 5
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
lm_model = TextClassification(model, tokenizer)


if __name__ == '__main__':
    # dm.prepare_data()
    # dm.setup()
    #
    # dm.train_dataloader()
    # for i, batch in enumerate(dm.train_dataloader()):
    #     train_loss = lm_model.training_step(batch, i)
    #     val_loss = lm_model.validation_step(batch, i)

    trainer = Trainer(max_steps=5)
    trainer.fit(lm_model, datamodule=dm)


