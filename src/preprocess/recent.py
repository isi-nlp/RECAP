import os
import json
import argparse
from collections import defaultdict



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path to the raw data", default="data/raw")
    parser.add_argument('-o', '--output_path', type=str, help="output processed data path", default="data/recent")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for split in ["train", "valid", "test", "pandora"]:
        lines = []
        user2lines = defaultdict(list)
        with open(os.path.join(args.data_path, f"{split}.jsonl")) as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                lines.append(line)

        lines = sorted(lines, key=lambda x:x[1])
        for i, line in enumerate(lines):
            author = line[0]
            user2lines[author].append(i)  

        authors = sorted(user2lines.keys())
        author2idx = {author: idx for idx, author in enumerate(authors)}
        
        author_comments = defaultdict(list)
        for author in authors:
            author_lines = user2lines.get(author, None)
            assert author_lines
            for author_line in author_lines:
                line = lines[author_line]
                _author, timestamp, src, tgt, subreddit = line
                assert author == _author
                author_comments[author].append((int(timestamp), src, tgt))
                author_comments[author] = sorted(author_comments[author], key=lambda x:x[0])
        assert len(author_comments) == len(authors)
        
        global_i = 0
        with open(os.path.join(args.output_path, f"{split}.jsonl"), 'w') as f:
            for author, samples in author_comments.items():
                recent_responses = []
                for i, (timestamp, src, tgt) in enumerate(samples):
                    ex = {
                        "id": global_i,
                        "author": author,
                        "author_id": author2idx[author],
                        "timestamp": timestamp,
                        "src": src,
                        "tgt": tgt,
                    }
                    ex["style_tgts"] = recent_responses
                    ex["semantic_tgts"] = None

                    if i >= 90:
                        global_i += 1
                        json_line = json.dumps(ex)
                        f.write(f"{json_line}\n")

                    recent_responses.insert(0, f"{tgt}")
                    if len(recent_responses) > 10:
                        del recent_responses[-1]