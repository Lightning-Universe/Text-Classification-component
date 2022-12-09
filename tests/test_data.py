

import torch

from lai_textclf.data import (
    TextEncodingCollate,
)

class BoringTokenizer:
    def __call__(self, texts, max_length: int = 256, **kwargs):
        return {"input_ids": torch.rand(len(texts), max_length)}


def test_text_encoding_collate():
    collate = TextEncodingCollate(BoringTokenizer(), 512)

    output = collate(
        [
            {"text": "this is cool!", "label": 3},
            {"text": "this is super cool!", "label": 5},
        ]
    )
    assert isinstance(output, dict)
    assert isinstance(output["input_ids"], torch.Tensor)
    assert output["input_ids"].shape == (2, 512)
    assert isinstance(output["labels"], torch.Tensor)
    assert output["labels"].shape == (2,)
