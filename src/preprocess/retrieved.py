import os
import json
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict

import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the raw data", default="data/raw")
    parser.add_argument('-r', '--retrieved_path', type=str, help="path to the retriever output", default="temp/retrieved")
    parser.add_argument('-o', '--output_path', type=str, help="output processed data path", default="data/retrieved")
    args = parser.parse_args()

    retrieved = {
        "style": torch.load(os.path.join(args.retrieved_path, "style.pt")),
        "semantic": torch.load(os.path.join(args.retrieved_path, "semantic.pt")),
    }

    for split in ["train", "valid", "test", "pandora"]:
        sorted_style = retrieved['style'][split]
        sorted_semantic = retrieved['semantic'][split]

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
            for author_line in author_lines:
                line = lines[author_line]
                _author, timestamp, src, tgt, subreddit = line
                assert author == _author
                author_comments[author].append((int(timestamp), src, tgt, author_line))
                author_comments[author] = sorted(author_comments[author], key=lambda x:x[0])
        assert len(author_comments) == len(authors)

        global_i = 0
        with open(os.path.join(args.output_path, f"{split}.jsonl"), 'w') as f:
            for i, (author, samples) in enumerate(tqdm(author_comments.items(), disable=True)):
                most_similar_style = sorted_style[i*10:(i+1)*10]
                most_similar_semantic = sorted_semantic[i*10:(i+1)*10]
                for i, (timestamp, src, tgt, _) in enumerate(samples[-10:]):
                    ex = {
                        "id": global_i,
                        "author": author,
                        "author_id": author2idx[author],
                        "timestamp": timestamp,
                        "src": src,
                        "tgt": tgt,
                    }
                    ex["style_tgts"] = [samples[j][2] for j in most_similar_style[i][:10]]
                    ex["semantic_tgts"] = [samples[j][2] for j in most_similar_semantic[i][:10]]

                    global_i += 1
                    json_line = json.dumps(ex)
                    f.write(f"{json_line}\n")
