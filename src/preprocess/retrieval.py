import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import random
import json

import torch
from datasets import load_dataset, load_from_disk


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the raw data", default="data/raw")
    parser.add_argument('-o', '--output_path', type=str, help="output processed data path", default="data/retrieval")
    args = parser.parse_args()

    for split in ["train", "valid", "test", "pandora"]:
        lines = []
        user2lines = defaultdict(list)
        with open(os.path.join(args.data_path, f"{split}.jsonl")) as f:
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

        with open(os.path.join(args.output_path, f"{split}.jsonl"), 'w') as f:
            for author, samples in tqdm(author_comments.items(), disable=True):

                ex = {
                    "srcs": [sample[1] for sample in samples],
                    "tgts": [sample[2] for sample in samples],
                }

                json_line = json.dumps(ex)
                f.write(f"{json_line}\n")