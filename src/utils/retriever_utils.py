from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy


def encode_example(example, tokenizer):
    max_len=128

    encoded_example = {}
    encoded_srcs = tokenizer(
        [txt.replace("<|TITLE|> ", "").replace(" <|EOS|> ", tokenizer.eos_token) for txt in example['srcs']],
        max_length=max_len//2,
        truncation=True,
        padding="max_length",
    )
    encoded_tgts = tokenizer(
        [txt for txt in example['tgts']],
        max_length=max_len//2, 
        truncation=True, 
        padding="max_length"
    )

    encoded_example = {
        "srcs_ids": encoded_srcs.input_ids,
        "srcs_attention_mask": encoded_srcs.attention_mask,
        "tgts_ids": encoded_tgts.input_ids,
        "tgts_attention_mask": encoded_tgts.attention_mask,
    }

    return encoded_example


@dataclass
class DataCollator:

    def __call__(self, features, return_tensors=None):
        import numpy as np

        seq_num = len(features[0]['srcs_ids'])
        features = [{k if 'rep' not in k else 'labels': 
                     torch.vstack(v) if 'rep' not in k else v.reshape(seq_num, -1) 
                     for k, v in feature.items()} for feature in features]
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.cat([feature[key].unsqueeze(0) for feature in features], dim=0)

        return batch