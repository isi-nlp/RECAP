import os
import shutil
import argparse

from utils.generator_utils import encode_example, DataCollator

from datasets import load_dataset, load_from_disk

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

if __name__ == "__main__":

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the dataset", default="data/recent")
    parser.add_argument('-s', '--save_path', type=str, help="path to save trained model", default="temp/generator")
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
        '-l', '--lr', 
        type=float, default=5e-5,
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, default=128,
    )
    parser.add_argument(
        '-g', '--grad_accumulation', 
        type=int, default=1,
    )
    parser.add_argument(
        '-w', '--warmup', 
        type=int, default=0,
    )
    parser.add_argument('--tqdm', action='store_true')
    args = parser.parse_args()

    task = f"{args.injection_mode}-{args.ref_type}"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    data_dir = args.data_path
    hf_dataset_path = os.path.join(data_dir, "hf_dataset")

    # tokenize raw data and save if needed
    if local_rank <= 0 and not os.path.exists(hf_dataset_path):
        print(f"tokenizing data in {data_dir} ...")

        cache_dir = os.path.join(data_dir, "cache")

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

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

        encoded_dataset = dataset.map(encode_example, num_proc=16, fn_kwargs={"tokenizer": tokenizer, "roberta_tokenizer": roberta_tokenizer})
        encoded_dataset.save_to_disk(hf_dataset_path)
        shutil.rmtree(cache_dir)
        del dataset, encoded_dataset
    
    # load hf dataset
    data_columns = [
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
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    else:
        model = PRG.from_pretrained("microsoft/DialoGPT-small")

    max_style_num = args.num_ref
    max_semantic_num = args.num_ref
    data_collator = DataCollator(
        tokenizer, 
        roberta_tokenizer,
        model,
        max_style_num=max_style_num,
        max_semantic_num=max_semantic_num,
    )

    save_path = args.save_path
    training_args = TrainingArguments(
        output_dir=f"{save_path}/model", 
        overwrite_output_dir=True,
        num_train_epochs=10, 
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.lr,
        label_smoothing_factor=0.0,
        warmup_steps=args.warmup,
        weight_decay=0.01, 
        fp16=True,
        logging_dir=f"{save_path}/log",
        evaluation_strategy="steps",
        logging_first_step=False,
        logging_steps=100,    # eval_steps is default to this value
        eval_steps=1000,
        save_steps=1000,
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

