# RECAP

The official repository for the ACL 2023 paper "Retrieval-Enhanced Context-Aware Prefix Encoder for Personalized Dialogue Response Generation".

# Installation

Commends for enviroment setup with conda.
```bash
conda create --name recap python=3.8
conda activate recap
pip install -U pip
pip install -r requirements.txt

```

# Data

The data is extracted from the Reddit dump from [pushshift.io](https://pushshift.io/). To preserve persona and personal writing style as much as possible, we did not filter out conversations with unethical content. You can download the raw data from the link [here](https://drive.google.com/file/d/1YC43Pqn15E7IIb90hjtauqRbwCOqAi3x/view?usp=sharing).

# Pre-processing

Pre-process the raw data into the format for retrieval and generation.

## Retrieval Data

### Encode text representations

```bash
python src/preprocess/encode_comments.py -d <raw_data_path> -o <output_path>
```

### Retrieval

```bash
python src/preprocess/retrieval.py -d <raw_data_path> -o <output_path>
```

## Generation Data

### Most recent hisotry responses

```bash
python src/preprocess/recent.py -d <raw_data_path> -o <output_path>
```

### Retrieved by hierarchical transformer

This requires the retriever output in `retrieved_path`. Please see section `training retriever` and `inference retrieve` for details on how to train and retrieve with the hierarchical transformer retriever.
```bash
python src/preprocess/retrieved.py -d <raw_data_path> -r <retrieved_path> -o <output_path>
```

# Training 

Train the retriever and the generator on a single GPU. The code works for multi GPUs, but the `batch_size` here is per device batch size, so please change it accordingly if you use more than one GPU.

## Retriever

```bash
python src/train_retriever.py \
    --data_path <data_path> \
    --raw_data_path <raw_data_path> \
    --reps_path <representations_path> \
    --save_path <save_path> \
    --ref_type <style OR semantic> \
    --lr 5e-5 \
    --batch_size 4 \
    --grad_accumulation 8 \
    --warmup 6250 \
    --nhead 12
```

## Generator

```bash
python src/train_generator.py \
    --data_path <data_path> \
    --save_path <save_path> \
    --injection_mode <(optional) concat OR context-prefix> \
    --ref_type <(optional) style OR semantic> \
    --lr 5e-5 \
    --batch_size 128 \
    --warmup 10000
```

# Inference

Retrieve and generate with trained models.

## Retrieve

```bash
python src/retrieve.py \
    --data_path <data_path> \
    --model_path <retriever_model_path> \
    --save_path <save_path> \
    --ref_type <style OR semantic>
```

## Generate

```bash
python src/generated.py \
    --data_path <data_path> \
    --model_path <generator_model_path> \
    --save_path <save_path> \
    --injection_mode <(optional) concat OR context-prefix> \
    --ref_type <(optional) style OR semantic>
```

## Evaluate

```bash
python src/eval.py \
    --generated_path <generated_responses_path> \
    --dataset_path <data_path> \
    --cache_dir <eval_cache_dir> \
    --cav_samples <eval_cav_samples_file>
```
