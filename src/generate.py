import os
import argparse
from collections import defaultdict
import json
from tqdm.auto import tqdm, trange

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    set_seed,
)

from datasets import load_from_disk

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy

from utils.generator_utils import encode_example, DataCollator, generate_responses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the dataset", default="data/recent")
    parser.add_argument('-m', '--model_path', type=str, help="path to trained model", default="temp/model")
    parser.add_argument('-s', '--save_path', type=str, help="path to save generated texts", default="temp/generated")
    parser.add_argument(
        '-i', '--injection_mode', 
        type=str, default=None, 
        choices=["concat", "prefix", "context-prefix"],
    )
    parser.add_argument(
        '-r', '--ref_type', 
        type=str, default=None, 
        choices=["style", "semantic", "mixed"],
    )
    parser.add_argument(
        '-n', '--num_ref', 
        type=int, default=10,
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, default=128,
    )
    parser.add_argument('--tqdm', action='store_true')
    args = parser.parse_args()

    task = f"{args.injection_mode}-{args.ref_type}"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    data_dir = args.data_path
    hf_dataset_path = os.path.join(data_dir, "hf_dataset")
    checkpoint_path = os.path.join(args.model_path, "checkpoint-best")
    
    # load hf dataset
    data_columns = [
        'id',
        'author_id',
        'input_ids', 
        'attention_mask', 
        'labels', 
    ]

    if args.injection_mode == "concat":
        assert args.ref_type != 'mixed', "ref_type == mixed only supports prefix based models"
        from models.dialogpt import GPT2LMHeadModel as PRG
        data_columns += [
            f'decoder_{args.ref_type}_ids', 
            f'decoder_{args.ref_type}_attention_mask',
        ]
    elif args.ref_type is not None:
        if args.injection_mode == "prefix":
            from models.dialogpt import PrefixGPT2 as PRG
        elif args.injection_mode == "context-prefix":
            from models.dialogpt import ContextPrefixGPT2 as PRG
            data_columns += ['context_ids', 'context_attention_mask']
        
        if args.ref_type != 'mixed':
            data_columns += [
                f'{args.ref_type}_ids', 
                f'{args.ref_type}_attention_mask',
            ]
        else:
            data_columns += [
                'style_ids', 
                'style_attention_mask',
                'semantic_ids', 
                'semantic_attention_mask',
            ]

    dataset = load_from_disk(f"{data_dir}/hf_dataset")
    print(f"data loaded from {data_dir}")
    
    dataset.set_format(
        type='torch', 
        columns=data_columns,
    )

    if args.injection_mode is None:
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path, local_files_only=True)
    else:
        model = PRG.from_pretrained(checkpoint_path, local_files_only=True)

    max_style_num = args.num_ref
    max_semantic_num = args.num_ref
    data_collator = DataCollator(
        tokenizer, 
        roberta_tokenizer,
        model,
        max_style_num=max_style_num,
        max_semantic_num=max_semantic_num,
    )
    
    lr = 5e-5
    training_args = TrainingArguments(
        save_strategy="no",
        output_dir="./temp/temp", 
        overwrite_output_dir=True,
        num_train_epochs=10, 
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size, 
        gradient_accumulation_steps=1,
        learning_rate=lr,
        label_smoothing_factor=0.0,
        warmup_steps=0,
        weight_decay=0.01, 
        fp16=False,
        logging_dir=None,
        evaluation_strategy="steps",
        logging_first_step=False,
        logging_steps=2000,    # eval_steps is default to this value
        save_steps=2000,
        save_total_limit=1,
        do_eval=True, 
        do_predict=False,
        metric_for_best_model="loss",
        load_best_model_at_end=False,
        disable_tqdm=not args.tqdm,
        group_by_length=False,
        length_column_name="length",
        dataloader_num_workers=16,
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

    set_seed(42)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for p_prob in [0.8]:
        generated = generate_responses(
            trainer,
            dataset["test"],
            max_length=128, 
            num_return_sequences=1, 
            do_sample=True,
            top_p=p_prob,
            top_k=0,
            disable_tqdm=not args.tqdm
        )
        
        with open(os.path.join(args.save_path, "generated.jsonl"), "w") as f:
            for ex in generated:
                f.write(json.dumps(ex) + '\n')
