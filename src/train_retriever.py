import os
import json
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy

from utils.retriever_utils import encode_example, DataCollator
from models.retriever import Retriever


if __name__ == "__main__":

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the dataset", default="data/retrieval")
    parser.add_argument('-u', '--raw_data_path', type=str, help="path to the raw data", default="data/raw")
    parser.add_argument('-e', '--reps_path', type=str, help="path to pre-encoded representations", default="temp/reps")
    parser.add_argument('-s', '--save_path', type=str, help="path to save trained model", default="temp/retriever")
    parser.add_argument(
        '-r', '--ref_type', 
        type=str, default="style", 
        choices=["style", "semantic"],
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, default=8,
    )
    parser.add_argument(
        '-g', '--grad_accumulation', 
        type=int, default=1,
    )
    parser.add_argument(
        '-l', '--lr', 
        type=float, default=5e-5,
    )
    parser.add_argument(
        '-w', '--warmup', 
        type=int, default=0,
    )
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument(
        '--nhead', 
        type=int, default=12,
    )
    args = parser.parse_args()
    
    data_dir = args.data_path
    hf_dataset_path = os.path.join(data_dir, "hf_dataset")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, local_files_only=True)

    # preprocess data and save if needed
    if local_rank <= 0 and not os.path.exists(hf_dataset_path):
        cache_dir = os.path.join(data_dir, "cache")
        # tokenize data
        print(f"tokenize data in {data_dir}")
        data_files = {
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "test": "test.jsonl",
        }
        dataset = load_dataset(
            data_dir, 
            data_files=data_files, 
            cache_dir=cache_dir,
        )
        encoded_dataset = dataset.map(encode_example, num_proc=16, fn_kwargs={"tokenizer": tokenizer})
        encoded_dataset.save_to_disk(f"{data_dir}/hf_dataset")
        shutil.rmtree(cache_dir)
        del dataset, encoded_dataset

        # add pre-encoded representation vectors
        reps = defaultdict(lambda: defaultdict(list))

        for split in ["train", "valid", "test"]:
            lines = []
            user2lines = defaultdict(list)
            with open(os.path.join(args.raw_data_path, f"{split}.jsonl")) as f:
                for i, line in tqdm(enumerate(f), disable=True):
                    line = json.loads(line)
                    lines.append(line)

            lines = sorted(lines, key=lambda x:x[1])
            for i, line in enumerate(lines):
                author = line[0]
                user2lines[author].append(i)  

            authors = sorted(user2lines.keys())
            author2idx = {author: idx for idx, author in enumerate(authors)}

            author_comments = defaultdict(list)
            for author in tqdm(authors, disable=True):
                author_lines = user2lines.get(author, None)
                assert author_lines
                prev_t = 0
                for author_line in author_lines:
                    line = lines[author_line]
                    _author, timestamp, src, tgt, subreddit = line

                    assert author == _author
                    assert prev_t <= timestamp
                    prev_t = timestamp

                    author_comments[author].append((int(timestamp), src, tgt, author_line))
                    author_comments[author] = sorted(author_comments[author], key=lambda x:x[0])
            assert len(author_comments) == len(authors)

            style_reps = torch.load(os.path.join(args.reps_path, f"style/{split}_fp16.pt"))['tgt']
            semantic_reps = torch.load(os.path.join(args.reps_path, f"semantic/{split}_fp16.pt"))['tgt']
            style_reps = torch.nn.functional.normalize(style_reps, dim=-1)
            semantic_reps = torch.nn.functional.normalize(semantic_reps, dim=-1)

            for author, samples in tqdm(author_comments.items(), disable=True):

                glb_ids = torch.tensor([sample[3] for sample in samples])
                tgt_style_reps = style_reps[glb_ids]
                tgt_semantic_reps = semantic_reps[glb_ids]     
                
                reps['style'][split].append(tgt_style_reps.cpu())
                reps['semantic'][split].append(tgt_semantic_reps.cpu())
            
        reps = {
            k0: {
                k1: [
                    item.flatten().float().numpy() for item in v1
                ] for k1, v1 in v0.items()
            } for k0, v0 in reps.items()
        }

        encoded_dataset = load_from_disk(f"{data_dir}/hf_dataset")

        for rep_type in reps.keys():
            for split in reps[rep_type].keys():
                print(rep_type, split)
                encoded_dataset[split] = encoded_dataset[split].add_column(f"{rep_type}_rep", reps[rep_type][split])
        
        shutil.rmtree(f"{data_dir}/hf_dataset")
        encoded_dataset.save_to_disk(f"{data_dir}/hf_dataset")

    data_columns = [
        'srcs_ids', 
        'srcs_attention_mask', 
        'tgts_ids', 
        'tgts_attention_mask',
        f'{args.ref_type}_rep',
    ]
    dataset = load_from_disk(f"{data_dir}/hf_dataset")
    print(f"data loaded from {data_dir}")
    dataset.set_format(
        type='torch', 
        columns=data_columns,
    )
    model = Retriever("distilroberta-base", nhead=args.nhead, use_gold_tgt_rep=False)

    data_collator = DataCollator()
    
    lr = args.lr
    save_path = args.save_path
    training_args = TrainingArguments(
        output_dir=f"{save_path}/{args.ref_type}/model", 
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=lr,
        label_smoothing_factor=0.0,
        warmup_steps=args.warmup,
        weight_decay=0.01, 
        fp16=True,
        logging_dir=f"{save_path}/{args.ref_type}/log",
        evaluation_strategy="steps",
        logging_first_step=False,
        logging_steps=125,    # eval_steps is default to this value
        eval_steps=625,
        save_steps=625,
        save_total_limit=1,
        do_eval=True, 
        do_predict=False,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        disable_tqdm=not args.tqdm,
        group_by_length=False,
        length_column_name="length",
        dataloader_num_workers=8,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"# Params: {pytorch_total_params}")

    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    print(trainer.train())
    print("finish training")
    print("evaluate on validation set")
    print(trainer.evaluate())
    print("evaluate on test set")
    print(trainer.evaluate(dataset["test"]))