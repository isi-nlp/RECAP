import os
import json
import argparse
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the raw data", default="data/raw")
    parser.add_argument('-o', '--output_path', type=str, help="output processed data path", default="temp/reps")
    args = parser.parse_args()

    models = {}
    models['style'] = SentenceTransformer('AnnaWegmann/Style-Embedding', device='cuda').half()
    models['semantic'] = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda').half()

    for split in ["train", "valid", "test", "pandora"]:
        lines = []
        with open(os.path.join(args.data_path, f"{split}.jsonl")) as f:
            for i, line in tqdm(enumerate(f), disable=True):
                line = json.loads(line)
                lines.append(line)
        lines = sorted(lines, key=lambda x:x[1])
        srcs = [line[2].replace("<|TITLE|> ", "").replace(" <|EOS|> ", "\n") for line in lines]
        tgts = [line[3] for line in lines]

        for rep_type in ['style', 'semantic']:
            reps = {}
            reps['src'] = models[rep_type].encode(srcs, batch_size=1024, convert_to_tensor=False, normalize_embeddings=False, show_progress_bar=False)
            reps['tgt'] = models[rep_type].encode(tgts, batch_size=1024, convert_to_tensor=False, normalize_embeddings=False, show_progress_bar=False)
            reps['src'] = torch.from_numpy(reps['src']).half()
            reps['tgt'] = torch.from_numpy(reps['tgt']).half()
            torch.save(reps, os.path.join(args.output_path, f"{rep_type}/{split}_fp16.pt"), create_dir=True)