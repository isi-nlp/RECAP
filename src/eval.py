import os
import json
import copy
import random
import argparse
import numpy as np
from tqdm.auto import trange, tqdm
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from ignite.metrics import RougeL
from nltk.translate.meteor_score import meteor_score
from bleurt import score as bleurt_score
from bert_score import score as compute_bert_score

import torch
from torch import nn

from transformers import set_seed
from datasets import load_from_disk

from sentence_transformers import SentenceTransformer


def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def corpus_meteor(refs, cands):
    scores = []
    for ref, cand in zip(refs, cands):
        scores.append(meteor_score(ref, cand))
    return np.mean(scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generated_path', type=str, required=True)
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-c', '--cache_dir', type=str, required=True)
    parser.add_argument('-v', '--cav_samples', type=str, required=True)
    args = parser.parse_args()

    # preprocess and save if needed
    if not os.path.exists(args.cache_dir):
        print("Encoding and preprocess labels")

        dataset = load_from_disk(args.dataset_path + '/test')

        generated = []
        with open(args.generated_path) as f:
            for line in f:
                generated.append(json.loads(line))

        txts = []
        for i in range(len(dataset)):
            txts.append(generated[i]["label"])

        assert len(generated) == len(txts)

        model = SentenceTransformer('AnnaWegmann/Style-Embedding', device='cuda')
        style_vecs = model.encode(txts, batch_size=512, show_progress_bar=True)
        model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        semantic_vecs = model.encode(txts, batch_size=512, show_progress_bar=True)

        author_txts = defaultdict(list)
        author_style_vecs = defaultdict(list)
        author_semantic_vecs = defaultdict(list)
        line2author = []
        for i in trange(len(dataset), disable=True):
            author_id = dataset[i]["author"]
            line2author.append(author_id)
            author_txts[author_id].append(txts[i])
            author_style_vecs[author_id].append(torch.tensor(style_vecs[i]))
            author_semantic_vecs[author_id].append(torch.tensor(semantic_vecs[i]))

        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

        torch.save(author_style_vecs, os.path.join(args.cache_dir, "style_vecs.pt"))
        torch.save(author_semantic_vecs, os.path.join(args.cache_dir, "semantic_vecs.pt"))
        with open(os.path.join(args.cache_dir, "txts.json"), "w") as f:
            json.dump(author_txts, f)
        with open(os.path.join(args.cache_dir, "line2author.json"), "w") as f:
            json.dump(line2author, f)

    # evaluate
    print("Evaluating")
    label_style_vecs = torch.load(os.path.join(args.cache_dir, "style_vecs.pt"))
    label_semantic_vecs = torch.load(os.path.join(args.cache_dir, "semantic_vecs.pt"))
    with open(os.path.join(args.cache_dir, "txts.json")) as f:
        label_txts = json.load(f)
    with open(args.cav_samples) as f:
        dc_cav_samples = json.load(f)
    author_ids = sorted(label_txts.keys())
    with open(os.path.join(args.cache_dir, "line2author.json")) as f:
        line2author = json.load(f)

    cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    set_seed(42)

    full_path = args.generated_path

    generated = []
    with open(full_path) as f:
        for line in f:
            generated.append(json.loads(line))

    txts = []
    for i in range(len(line2author)):
        txts.append(generated[i]["generated"][0])

    model = SentenceTransformer('AnnaWegmann/Style-Embedding', device='cuda')
    style_vecs = model.encode(txts, batch_size=128, show_progress_bar=True)
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    semantic_vecs = model.encode(txts, batch_size=128, show_progress_bar=True)

    gen_txts = defaultdict(list)
    gen_style_vecs = defaultdict(list)
    gen_semantic_vecs = defaultdict(list)

    for i in trange(len(line2author), disable=True):
        author_id = line2author[i]
        gen_txts[author_id].append(txts[i])
        gen_style_vecs[author_id].append(torch.tensor(style_vecs[i]))
        gen_semantic_vecs[author_id].append(torch.tensor(semantic_vecs[i]))

    combined_txts = []
    combined_refs = []

    for author_id in tqdm(author_ids, disable=True):

        combined_refs += label_txts[author_id]
        combined_txts += gen_txts[author_id]

    # BLEURT
    scorer = bleurt_score.BleurtScorer('BLEURT-20-D3') # please download the bleurt checkpoint from https://github.com/google-research/bleurt
    bleurt_result = scorer.score(references=combined_refs, candidates=combined_txts)
    print(f"BELURT: {np.mean(bleurt_result):.4f}")

    # BERTScore
    print("Computing BERTScore...")
    (P, R, F), hashname = compute_bert_score(combined_txts, combined_refs, lang="en", return_hash=True)
    print(f"BERTScore: {float(F.mean()):.4f}")

    # BLEU
    combined_txts = [word_tokenize(item) for item in tqdm(combined_txts, disable=True)]
    combined_refs = [[word_tokenize(item)] for item in tqdm(combined_refs, disable=True)]

    bleu_1 = corpus_bleu(combined_refs, combined_txts, weights=(1.0,))
    bleu_2 = corpus_bleu(combined_refs, combined_txts, weights=(0.5, 0.5))
    bleu_3 = corpus_bleu(combined_refs, combined_txts, weights=(1./3, 1./3, 1./3))
    bleu_4 = corpus_bleu(combined_refs, combined_txts, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-(1|2|3|4): {bleu_1:.4f} | {bleu_2:.4f} | {bleu_3:.4f} | {bleu_4:.4f}")

    # rouge-l
    rouge_l = RougeL()
    rouge_l.update((combined_txts, combined_refs))
    rouge_l_score = rouge_l.compute()
    print(f"ROUGE-L: {rouge_l_score['Rouge-L-F']:.4f}")

    # meteor
    meteor = corpus_meteor(combined_refs, combined_txts)
    print(f"METEOR: {meteor:.4f}")

    # style CAV (w/ domain control)
    acc = torch.zeros(1)
    total = 0

    for author_id in tqdm(author_ids, disable=True):
        if author_id in dc_cav_samples:
            a_vecs = []
            p_vecs = []
            n_vecs = []
            author_samples = dc_cav_samples[author_id]
            for a_id, p_id, neg_author, n_id, _, _ in author_samples:
                assert neg_author != author_id
                a_vecs.append(gen_style_vecs[author_id][a_id])
                p_vecs.append(label_style_vecs[author_id][p_id])
                n_vecs.append(label_style_vecs[neg_author][n_id])
            a_vecs = torch.vstack(a_vecs)
            p_vecs = torch.vstack(p_vecs)
            n_vecs = torch.vstack(n_vecs)

            sp_cossim = cossim(a_vecs, p_vecs)
            sn_cossim = cossim(a_vecs, n_vecs)
            acc += (sp_cossim > sn_cossim).float().sum()
            
            total += len(author_samples)
    style_acc = float(acc / total)

    # semantic CAV (w/ domain control)
    acc = torch.zeros(1)
    total = 0

    for author_id in tqdm(author_ids, disable=True):
        if author_id in dc_cav_samples:
            a_vecs = []
            p_vecs = []
            n_vecs = []
            author_samples = dc_cav_samples[author_id]
            for a_id, p_id, neg_author, n_id, _, _ in author_samples:
                assert neg_author != author_id
                a_vecs.append(gen_semantic_vecs[author_id][a_id])
                p_vecs.append(label_semantic_vecs[author_id][p_id])
                n_vecs.append(label_semantic_vecs[neg_author][n_id])
            a_vecs = torch.vstack(a_vecs)
            p_vecs = torch.vstack(p_vecs)
            n_vecs = torch.vstack(n_vecs)

            sp_cossim = cossim(a_vecs, p_vecs)
            sn_cossim = cossim(a_vecs, n_vecs)
            acc += (sp_cossim > sn_cossim).float().sum()
            
            total += len(author_samples)
    semantic_acc = float(acc / total)

    print(f"CAV accuracy (style | semantic): {style_acc:.4f} | {semantic_acc:.4f}")

    # style embedding similarity
    sum_cossim = torch.zeros(1)

    for author_id in tqdm(author_ids, disable=True):
        mean_cossim = cossim(torch.vstack(label_style_vecs[author_id]), torch.vstack(gen_style_vecs[author_id])).mean()
        sum_cossim += mean_cossim
    style_cossim = float(sum_cossim / len(author_ids))
    
    # semantic embedding similarity
    sum_cossim = torch.zeros(1)

    for author_id in tqdm(author_ids, disable=True):
        mean_cossim = cossim(torch.vstack(label_semantic_vecs[author_id]), torch.vstack(gen_semantic_vecs[author_id])).mean()
        sum_cossim += mean_cossim
    semantic_cossim = float(sum_cossim / len(author_ids))

    print(f"Embedding Similarity (style | semantic): {style_cossim:.4f} | {semantic_cossim:.4f}")