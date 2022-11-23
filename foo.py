from lai_textclf.data import TextClassificationDataModule
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dm = TextClassificationDataModule('YelpReviewFull', tokenizer)



if __name__ == '__main__':
    dm.prepare_data()
    dm.setup()

    dm.train_dataloader()
    for i, batch in enumerate(dm.train_dataloader()):
        print(batch)
        if i > 10:
            break
