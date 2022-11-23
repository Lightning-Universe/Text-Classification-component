# MIT License
#
# Copyright (c) 2021 Shivanand Roy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Code in this file is based on https://github.com/Shivanandroy/simpleT5 by Shivanand Roy
"""


#
# class TextSummarization(LightningModule):
#     """PyTorch Lightning Model class"""
#
#     def __init__(
#         self,
#         model,
#         tokenizer,
#     ):
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.average_training_loss = None
#         self.average_validation_loss = None
#         self.save_only_last_epoch = False
#
#     def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
#         """forward step"""
#         output = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             decoder_attention_mask=decoder_attention_mask,
#         )
#
#         return output.loss, output.logits
#
#     def training_step(self, batch, batch_size):
#         """training step"""
#         input_ids = batch["source_text_input_ids"]
#         attention_mask = batch["source_text_attention_mask"]
#         labels = batch["labels"]
#         labels_attention_mask = batch["labels_attention_mask"]
#
#         loss, outputs = self(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_attention_mask=labels_attention_mask,
#             labels=labels,
#         )
#
#         self.log(
#             "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
#         )
#         return loss
#
#     def validation_step(self, batch, batch_size):
#         """validation step"""
#         input_ids = batch["source_text_input_ids"]
#         attention_mask = batch["source_text_attention_mask"]
#         labels = batch["labels"]
#         labels_attention_mask = batch["labels_attention_mask"]
#
#         loss, outputs = self(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_attention_mask=labels_attention_mask,
#             labels=labels,
#         )
#
#         self.log(
#             "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
#         )
#         return loss
#
#     def test_step(self, batch, batch_size):
#         """test step"""
#         input_ids = batch["source_text_input_ids"]
#         attention_mask = batch["source_text_attention_mask"]
#         labels = batch["labels"]
#         labels_attention_mask = batch["labels_attention_mask"]
#
#         loss, outputs = self(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_attention_mask=labels_attention_mask,
#             labels=labels,
#         )
#
#         self.log("test_loss", loss, prog_bar=True, logger=True)
#         return loss
#
#     def configure_optimizers(self):
#         """configure optimizers"""
#         return AdamW(self.parameters(), lr=0.0001)
#
#     def validation_epoch_end(self, validation_step_outputs):
#         _loss = [x.cpu() for x in validation_step_outputs]
#         self.average_validation_loss = np.round(
#             torch.mean(torch.stack(_loss)).item(),
#             4,
#         )
#
#
# def predict(
#     module: LightningModule,
#     source_text: str,
#     max_length: int = 512,
#     num_return_sequences: int = 1,
#     num_beams: int = 2,
#     top_k: int = 50,
#     top_p: float = 0.95,
#     do_sample: bool = True,
#     repetition_penalty: float = 2.5,
#     length_penalty: float = 1.0,
#     early_stopping: bool = True,
#     skip_special_tokens: bool = True,
#     clean_up_tokenization_spaces: bool = True,
# ):
#     """
#     generates prediction for T5/MT5 model
#     Args:
#         source_text (str): any text for generating predictions
#         max_length (int, optional): max token length of prediction. Defaults to 512.
#         num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
#         num_beams (int, optional): number of beams. Defaults to 2.
#         top_k (int, optional): Defaults to 50.
#         top_p (float, optional): Defaults to 0.95.
#         do_sample (bool, optional): Defaults to True.
#         repetition_penalty (float, optional): Defaults to 2.5.
#         length_penalty (float, optional): Defaults to 1.0.
#         early_stopping (bool, optional): Defaults to True.
#         skip_special_tokens (bool, optional): Defaults to True.
#         clean_up_tokenization_spaces (bool, optional): Defaults to True.
#     Returns:
#         list[str]: returns predictions
#     """
#     input_ids = module.tokenizer.encode(
#         source_text, return_tensors="pt", add_special_tokens=True
#     )
#     input_ids = input_ids.to(module.device)
#     generated_ids = module.model.generate(
#         input_ids=input_ids,
#         num_beams=num_beams,
#         max_length=max_length,
#         repetition_penalty=repetition_penalty,
#         length_penalty=length_penalty,
#         early_stopping=early_stopping,
#         top_p=top_p,
#         top_k=top_k,
#         num_return_sequences=num_return_sequences,
#     )
#     preds = [
#         module.tokenizer.decode(
#             g,
#             skip_special_tokens=skip_special_tokens,
#             clean_up_tokenization_spaces=clean_up_tokenization_spaces,
#         )
#         for g in generated_ids
#     ]
#     return preds
