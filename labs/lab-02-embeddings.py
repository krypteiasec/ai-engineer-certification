#!/usr/bin/env python3
"""
LAB 02: Embeddings: giving each token a vector of meaning.

A token id is just an integer, and 7 is not "more" than 3 in any meaningful
way. So we give every token a small vector of numbers, called an embedding. The
model learns these numbers so that tokens used in similar ways end up with
similar vectors. In this lab you build the embedding table with PyTorch's real
nn.Embedding (the exact layer GPT uses), look tokens up, and measure how similar
two token vectors are with cosine similarity.

Run: python3 modules/academy-content/labs/lab-02-embeddings.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)  # reproducible: same table every run

VOCAB = ["cat", "dog", "run", "walk", "the", "a"]  # 6 toy tokens
D = 4  # embedding dimension: each token becomes 4 numbers
stoi = {t: i for i, t in enumerate(VOCAB)}

# STEP 1: the embedding table is exactly nn.Embedding(num_tokens, D): a
# len(VOCAB) x D matrix of LEARNABLE numbers. Real models start these random and
# learn them during training. Here they start random and stay random (untrained).
table = nn.Embedding(len(VOCAB), D)


def embed(token):
    if token not in stoi:
        raise ValueError("token not in vocab: %s" % token)
    idx = torch.tensor(stoi[token], dtype=torch.long)
    return table(idx)  # the lookup IS the embedding: row `idx` of the table


print("STEP 1: the embedding table (one row per token, D=%d)" % D)
for i, t in enumerate(VOCAB):
    row = table.weight[i].tolist()
    print("  %-5s -> [%s]" % (t, ", ".join("%.2f" % n for n in row)))
print("")

# STEP 2: embedding a sequence is just stacking the row lookups. nn.Embedding
# does the whole sequence in one call when you pass it a tensor of ids.
seq = ["the", "cat", "run"]
seq_ids = torch.tensor([stoi[t] for t in seq], dtype=torch.long)
seq_vecs = table(seq_ids)  # shape (3, D) in one shot
print("STEP 2: embed a sequence (each token becomes its vector)")
print("  %s  ->  tensor shape %s" % (" ".join(seq), tuple(seq_vecs.shape)))
for t, v in zip(seq, seq_vecs):
    print("    %-5s [%s]" % (t, ", ".join("%.2f" % n for n in v.tolist())))
print("")

# STEP 3: cosine similarity measures the ANGLE between two vectors, from -1
# (opposite) to 1 (same direction). This is how we ask "are these two tokens
# used similarly?" once the table is trained. F.cosine_similarity is the real one.
print("STEP 3: cosine similarity between token vectors (untrained, so random)")
for x, y in [("cat", "dog"), ("cat", "run"), ("the", "a")]:
    cos = F.cosine_similarity(embed(x), embed(y), dim=0).item()
    print("  cos(%s, %s) = %.3f" % (x, y, cos))
print("")

# STEP 4: the invariant that makes embeddings a MODEL, not a lookup table: a
# token is always similar to ITSELF. cos(v, v) must equal 1 for every token.
self_ok = True
for t in VOCAB:
    v = embed(t)
    if abs(F.cosine_similarity(v, v, dim=0).item() - 1.0) > 1e-5:
        self_ok = False
print("STEP 4: every token is maximally similar to itself (cos == 1): %s" % ("YES" if self_ok else "NO"))
if not self_ok:
    sys.exit(1)
print("")
print("Untrained these numbers are random. In Chapter 8 you TRAIN a table like this and")
print("watch related tokens drift together. Next: use these vectors to make a prediction.")
