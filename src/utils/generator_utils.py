from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


def encode_example(example, tokenizer, roberta_tokenizer):
    max_len=128
    max_ref_len = {
        "style": 64,
        "semantic": 64,
    }

    src = tokenizer.eos_token + example['src'].replace("<|TITLE|> ", "").replace(" <|EOS|> ", tokenizer.eos_token)
    tgt = tokenizer.eos_token + example['tgt'] + tokenizer.eos_token
    srctgt = src + tgt + tokenizer.eos_token
    author = example.get('author_id', -1)
    assert isinstance(author, int)
    encoded_src = tokenizer(src)
    encoded_tgt = tokenizer(tgt)
    src_ids = encoded_src.input_ids
    tgt_ids = encoded_tgt.input_ids

    encoded_context = roberta_tokenizer(
        example['src'].replace("<|TITLE|> ", "").replace(" <|EOS|> ", roberta_tokenizer.eos_token)
    )
    context_ids = encoded_context.input_ids[-max_len // 2:]
    context_attention_mask = [1] * len(context_ids)

    srctgt_ids = src_ids + tgt_ids
    if (len(src_ids) + len(tgt_ids)) > max_len:
        if len(tgt_ids) > max_len:
            srctgt_ids = tgt_ids[:max_len]
        else:
            srctgt_ids = srctgt_ids[-max_len:]
    label_len = min(max_len, len(tgt_ids))
    input_ids = srctgt_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * (len(srctgt_ids) - label_len) + tgt_ids[:max_len]
    assert len(attention_mask) == len(input_ids)
    assert len(labels) == len(input_ids)

    encoded_example = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "author_labels": author,
        "context_ids": context_ids,
        "context_attention_mask": context_attention_mask,
    }

    ref_src = {
        "style": example['style_tgts'],
        "semantic": example['semantic_tgts'],
    }
    for ref_type, ref in ref_src.items():
        if ref is not None:
            # tokenize with roberta tokenizer
            encoded_ref = [roberta_tokenizer(item) for item in ref]
            ref_ids = [item.input_ids for item in encoded_ref]
            ref_attention_mask = [item.attention_mask for item in encoded_ref]
            for i in range(len(ref_ids)):
                if max_ref_len[ref_type] and len(ref_ids[i]) > max_ref_len[ref_type]:
                    ref_ids[i] = [ref_ids[i][0]] + ref_ids[i][1:max_ref_len[ref_type]-1] + [ref_ids[i][-1]]
                    ref_attention_mask[i] = [ref_attention_mask[i][0]] + ref_attention_mask[i][1:max_ref_len[ref_type]-1] + [ref_attention_mask[i][-1]]
            encoded_example[f"{ref_type}_ids"] = ref_ids
            encoded_example[f"{ref_type}_attention_mask"] = ref_attention_mask

            # tokenize with dialogpt tokenizer
            encoded_ref = [tokenizer(tokenizer.eos_token + item) for item in ref]
            ref_ids = [item.input_ids for item in encoded_ref]
            ref_attention_mask = [item.attention_mask for item in encoded_ref]
            for i in range(len(ref_ids)):
                if max_ref_len[ref_type] and len(ref_ids[i]) > max_ref_len[ref_type]:
                    ref_ids[i] = ref_ids[i][:max_ref_len[ref_type]]
                    ref_attention_mask[i] = ref_attention_mask[i][:max_ref_len[ref_type]]
            encoded_example[f"decoder_{ref_type}_ids"] = ref_ids
            encoded_example[f"decoder_{ref_type}_attention_mask"] = ref_attention_mask

    return encoded_example


@dataclass
class DataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    roberta_tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    max_style_num: int = 10
    max_semantic_num: int = 10

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors

        max_ref_num = {
            "style": self.max_style_num,
            "semantic": self.max_semantic_num,
        }

        decoder_ref_type = None
        if "decoder_style_ids" in features[0].keys():
            assert not "decoder_semantic_ids" in features[0].keys()
            decoder_ref_type = 'style'
        elif "decoder_semantic_ids" in features[0].keys():
            decoder_ref_type = 'semantic'
        if decoder_ref_type is not None:
            for feature in features:
                feature[f"decoder_{decoder_ref_type}_ids"] = np.concatenate(feature[f"decoder_{decoder_ref_type}_ids"][:max_ref_num[decoder_ref_type]])
                feature[f"decoder_{decoder_ref_type}_attention_mask"] = np.concatenate(feature[f"decoder_{decoder_ref_type}_attention_mask"][:max_ref_num[decoder_ref_type]])
                feature["input_ids"] = np.concatenate([feature[f"decoder_{decoder_ref_type}_ids"], feature["input_ids"]])
                feature["attention_mask"] = np.concatenate([feature[f"decoder_{decoder_ref_type}_attention_mask"], feature["attention_mask"]])
                if "labels" in features[0].keys():
                    feature["labels"] = np.concatenate([[-100] * len(feature[f"decoder_{decoder_ref_type}_ids"]), feature["labels"]])
                del feature[f"decoder_{decoder_ref_type}_ids"]
                del feature[f"decoder_{decoder_ref_type}_attention_mask"]
    
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        context_ids = [feature["context_ids"] for feature in features] if "context_ids" in features[0].keys() else None
        ref_ids = {
            ref_type: 
            [feature[f"{ref_type}_ids"] for feature in features] if f"{ref_type}_ids" in features[0].keys() else None
            for ref_type in ["style", "semantic"]
        }
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            assert padding_side == 'right'
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        if context_ids is not None:
            max_context_length = max(len(c) for c in context_ids)
            for feature in features:
                remainder = [self.roberta_tokenizer.pad_token_id] * (max_context_length - len(feature["context_ids"]))
                attn_mask_remainder = [0] * (max_context_length - len(feature["context_attention_mask"]))
                feature["context_ids"] = np.concatenate([feature["context_ids"], remainder]).astype(np.int64)
                feature["context_attention_mask"] = np.concatenate([feature["context_attention_mask"], attn_mask_remainder]).astype(np.int64)

        # get max length of all ref
        max_ref_length = 0
        for ref_type, type_ids in ref_ids.items():
            if type_ids is None:
                continue
            max_type_num = min(max(len(hs) for hs in type_ids), max_ref_num[ref_type])
            max_type_length = max(len(h) for hs in type_ids for type_i, h in enumerate(hs) if type_i <= max_type_num)
            max_ref_length = max(max_ref_length, max_type_length)

        for ref_type, type_ids in ref_ids.items():
            if type_ids is None:
                continue
            max_type_num = min(max(len(hs) for hs in type_ids), max_ref_num[ref_type])
            max_type_length = max_ref_length

            for feature in features:
                if len(feature[f"{ref_type}_ids"]) < max_type_num:
                    remainder = [[self.roberta_tokenizer.pad_token_id]] * (max_type_num - len(feature[f"{ref_type}_ids"]))
                    attn_mask_remainder = [[0]] * (max_type_num - len(feature[f"{ref_type}_ids"]))
                    if isinstance(feature[f"{ref_type}_ids"], list):
                        feature[f"{ref_type}_ids"] += remainder
                        feature[f"{ref_type}_attention_mask"] += attn_mask_remainder
                    else:
                        assert False
                if len(feature[f"{ref_type}_ids"]) > max_type_num:
                    feature[f"{ref_type}_ids"] = feature[f"{ref_type}_ids"][:max_type_num]
                    feature[f"{ref_type}_attention_mask"] = feature[f"{ref_type}_attention_mask"][:max_type_num]
                
                for type_i in range(max_type_num):
                    remainder = [self.roberta_tokenizer.pad_token_id] * (max_type_length - len(feature[f"{ref_type}_ids"][type_i]))
                    attn_mask_remainder = [0] * (max_type_length - len(feature[f"{ref_type}_attention_mask"][type_i]))
                    if isinstance(feature[f"{ref_type}_ids"][type_i], list):
                        feature[f"{ref_type}_ids"][type_i] += remainder
                        feature[f"{ref_type}_attention_mask"][type_i] += attn_mask_remainder
                    else:
                        feature[f"{ref_type}_ids"][type_i] = np.concatenate(
                            [feature[f"{ref_type}_ids"][type_i], remainder]
                        ).astype(np.int64)
                        feature[f"{ref_type}_attention_mask"][type_i] = np.concatenate(
                            [feature[f"{ref_type}_attention_mask"][type_i], attn_mask_remainder]
                        ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if "style_ids" in features and "semantic_ids" in features:
            features["ref_input_ids"] = torch.cat((features["style_ids"][:, :5, :], features["semantic_ids"][:, :5, :]), 1)
            features["ref_attention_mask"] = torch.cat((features["style_attention_mask"][:, :5, :], features["semantic_attention_mask"][:, :5, :]), 1)
        else:
            ref_type = None
            if "style_ids" in features:
                ref_type = 'style'
            elif "semantic_ids" in features:
                ref_type = 'semantic'
            if ref_type is not None:
                features["ref_input_ids"] = features[f"{ref_type}_ids"]
                features["ref_attention_mask"] = features[f"{ref_type}_attention_mask"]
        
        if f"context_ids" in features:
            features["context_input_ids"] = features["context_ids"]
            del features["context_ids"]

        for key in list(features.keys()):
            if (key.startswith("style") or key.startswith("semantic")) and "ref" not in key:
                del features[key]

        return features


def generate_responses(trainer,
                       dataset,
                       max_length=128,
                       disable_tqdm=False,
                       out_file=None,
                       **kwargs):
    
    trainer.model.eval()

    original_per_device_eval_batch_size = trainer.args.per_device_eval_batch_size
    trainer.args.per_device_eval_batch_size = original_per_device_eval_batch_size // kwargs.get("num_beams", 1)

    tokenizer = trainer.tokenizer

    ids = []
    authors = []
    input_strs = []
    label_strs = []
    generated_strs = []
    
    for batch in tqdm(trainer.get_test_dataloader(dataset), disable=disable_tqdm):
        batch_ids = batch['id'].tolist()
        batch_authors = batch['author_id'].tolist()
        del batch['id']
        del batch['author_id']
        batch_ids = [-1] * len(batch['input_ids'])
        batch_authors = [-1] * len(batch['input_ids'])
        batch_labels = batch['labels']
        
        batch_gen_input_ids = []
        batch_gen_attention_mask = []
        max_input_len = 0
        
        for input_ids, attention_mask, labels in zip(batch['input_ids'], batch['attention_mask'], batch['labels']):
            gen_input_ids = input_ids[(attention_mask == 1) & (labels == -100)].tolist() + [50256]
            gen_attention_mask = [1] * len(gen_input_ids)
            batch_gen_input_ids.append(gen_input_ids)
            batch_gen_attention_mask.append(gen_attention_mask)
            max_input_len = max(max_input_len, len(gen_input_ids))
        for i in range(len(batch_gen_input_ids)):
            input_len = len(batch_gen_input_ids[i])
            if input_len < max_input_len:
                batch_gen_input_ids[i] = [50256] * (max_input_len - input_len) + batch_gen_input_ids[i]
                batch_gen_attention_mask[i] = [0] * (max_input_len - input_len) + batch_gen_attention_mask[i]
        batch['input_ids'] = torch.tensor(batch_gen_input_ids)
        batch['attention_mask'] = torch.tensor(batch_gen_attention_mask)
        
        batch_input_strs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        batch_input_strs = [item.replace('<|endoftext|>', '\n').strip() for item in batch_input_strs]
            
        batch_labels[batch_labels == -100] = tokenizer.pad_token_id
        
        batch_label_strs = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
        batch = {arg: val.to(trainer.args.device) if not isinstance(val, list) else [item.to(trainer.args.device) for item in val]
                for arg, val in batch.items()
                if arg not in ['author_id', 'labels', 'decoder_input_ids']}
        assert max_input_len == batch['input_ids'].size(1)
        batch_generated = trainer.model.generate(**batch, 
                                                 max_length=max_length + max_input_len,
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 **kwargs)
        batch_generated = batch_generated[:, max_input_len:]
        batch_generated_strs = tokenizer.batch_decode(batch_generated, skip_special_tokens=True)
        batch_generated_strs = [batch_generated_strs[i*1:(i+1)*1] 
                                for i in range(len(batch_generated_strs)//1)]

        authors += batch_authors
        input_strs += batch_input_strs
        label_strs += batch_label_strs
        generated_strs += batch_generated_strs

    generated_results = []
    for idx, author, input_str, label_str, generated_str in zip(ids, authors, input_strs, label_strs, generated_strs):
        generated_results.append(
            {
                "id" : idx,
                "author": author,
                "input": input_str,
                "label": label_str,
                "generated": generated_str
            }
        )
        
    trainer.args.per_device_eval_batch_size = original_per_device_eval_batch_size

    return generated_results