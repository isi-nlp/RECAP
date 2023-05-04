import os
import argparse
import numpy as np
from tqdm import trange
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_from_disk
from models.retriever import Retriever


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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the dataset", default="data/retrieval")
    parser.add_argument('-m', '--model_path', type=str, help="path to the retrieval model", default="temp/retriever")
    parser.add_argument('-s', '--save_path', type=str, help="path to save the output", default="temp/retrieved")
    parser.add_argument(
        '-r', '--ref_type', 
        type=str, default="style", 
        choices=["style", "semantic"],
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, default=2,
    )
    args = parser.parse_args()

    dataset = load_from_disk(os.path.join(args.data_path, "hf_dataset"))
    data_columns = [
        'srcs_ids', 
        'srcs_attention_mask',
        'tgts_ids', 
        'tgts_attention_mask',
        f'{args.ref_type}_rep',
    ]
    dataset.set_format(
        type='torch', 
        columns=data_columns,
    )

    data_collator = DataCollator()

    model = Retriever("distilroberta-base", use_gold_tgt_rep=False, nhead=12)

    path = os.path.join(args.model_path, args.ref_type, "model/checkpoint-best/pytorch_model.bin")
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    _ = model.cuda()
    _ = model.eval()

    bsz = args.batch_size
    retrieved_ids = defaultdict(list)
    for split in ['train', 'valid', 'test']:

        pbar = trange(0, len(dataset[split]), bsz, disable=False)
        for i in pbar:
            features = [dataset[split][j] for j in range(i, min(i+bsz, len(dataset[split])))]
            features = data_collator(features)
            batch = {k: v.cuda() for k, v in features.items()}

            preds = model(**batch).logits
            preds = F.normalize(preds, dim=-1)
            reps = F.normalize(batch['labels'], dim=-1)

            pred_sims = torch.matmul(preds, reps.transpose(1, 2))
            pred_sims += -2. * torch.ones_like(pred_sims).triu(diagonal=90)
            pred_sorted_sims, pred_sorted_indices = pred_sims.sort(descending=True, dim=-1)
            retrieved_ids[split].append(pred_sorted_indices[:, :, :10].cpu())

        retrieved_ids[split] = torch.cat(retrieved_ids[split], dim=0).reshape(1, -1, 10).squeeze(0)
    
    torch.save(retrieved_ids, os.path.join(args.save_path, f"{args.ref_type}.pt"))