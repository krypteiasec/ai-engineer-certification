#!/usr/bin/env python3
"""
LAB TR1: Datasets, the raw material a model learns from.

Before a model can learn anything, you have to turn text into the exact shape a
training loop consumes: a long stream of token ids, cut into (input, target)
pairs where the target is simply the NEXT token, then split into a training set
and a held-out validation set the model never trains on. Data quality and data
plumbing decide more outcomes than model size does, so this is where every real
fine-tune starts.

This lab builds that pipeline by hand:
  1. tokenize a tiny corpus into ids with the same CharTokenizer the Academy uses,
  2. cut a contiguous stream into (x, y) pairs where y is x shifted by one token,
  3. split disjointly into train and validation,
  4. write a get_batch dataloader that samples real training batches,
then PROVES the pipeline is well formed: every target is the next token, the two
splits never overlap, and every batch has the right shape.

Run: python3 modules/academy-content/labs/training/tr1-datasets.py
"""
import sys, os
_labs = None
for _c in [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
           os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]:
    if _c and os.path.isdir(os.path.join(_c, "_models")): _labs = os.path.abspath(_c); break
if _labs: sys.path.insert(0, os.path.join(_labs, "_models"))

import torch
from model import CharTokenizer  # the same char tokenizer the trained models use

# A tiny corpus. Real fine-tunes use megabytes; the plumbing is identical.
CORPUS = (
    "a model learns patterns from data. a good dataset is clean and consistent. "
    "we split data into a training set and a validation set. the model trains on "
    "the training set and we measure it on the validation set it never saw. "
) * 8
BLOCK = 16   # context length: how many tokens the model sees at once
BATCH = 8


def build_dataset(text):
    """Tokenize the text and return the id stream plus the tokenizer."""
    tok = CharTokenizer(sorted(set(text)))
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return tok, ids


def make_pairs(ids, block):
    """Every window of length block is an input; its target is the SAME window
    shifted one token to the right, so target[t] is the token that follows
    input[t]. That single idea (predict the next token) is all of language-model
    training."""
    x, y = [], []
    for i in range(len(ids) - block):
        x.append(ids[i:i + block])
        y.append(ids[i + 1:i + 1 + block])
    return torch.stack(x), torch.stack(y)


def split_train_val(ids, frac=0.9):
    """Disjoint split. The model trains on the first part and is judged on the
    second part it never trained on. No overlap is the whole point: overlap is
    how you fool yourself into thinking a model generalized."""
    n = int(len(ids) * frac)
    return ids[:n], ids[n:]


def get_batch(pairs_x, pairs_y, batch, gen):
    """A dataloader: sample `batch` random (x, y) pairs for one training step."""
    idx = torch.randint(0, pairs_x.shape[0], (batch,), generator=gen)
    return pairs_x[idx], pairs_y[idx]


def main():
    gen = torch.Generator().manual_seed(1234)
    tok, ids = build_dataset(CORPUS)
    print("STEP 1: tokenize the corpus")
    print("  corpus chars: %d | vocab: %d | token ids: %d" % (len(CORPUS), tok.vocab_size, len(ids)))

    train_ids, val_ids = split_train_val(ids)
    print("")
    print("STEP 2: disjoint train / validation split")
    print("  train tokens: %d | val tokens: %d" % (len(train_ids), len(val_ids)))

    xt, yt = make_pairs(train_ids, BLOCK)
    xv, yv = make_pairs(val_ids, BLOCK)
    print("")
    print("STEP 3: cut into next-token (input, target) pairs")
    print("  train pairs: %d | val pairs: %d | each shape: (%d,)" % (xt.shape[0], xv.shape[0], BLOCK))

    bx, by = get_batch(xt, yt, BATCH, gen)
    print("")
    print("STEP 4: the dataloader samples a real batch")
    print("  batch x shape: %s | batch y shape: %s" % (tuple(bx.shape), tuple(by.shape)))

    # ---- prove the pipeline is well formed ----
    # (a) target is the next token: for every pair, y[:-1] must equal x[1:].
    targets_are_next = bool(torch.equal(by[:, :-1], bx[:, 1:]))
    # also check it on the raw stream directly.
    stream_ok = bool(torch.equal(yt[:, :-1], xt[:, 1:]))
    # (b) the two splits never share a token position (disjoint by construction).
    n_train = len(train_ids)
    splits_disjoint = (n_train + len(val_ids) == len(ids)) and n_train > 0 and len(val_ids) > 0
    # (c) batches have the requested shape.
    shape_ok = tuple(bx.shape) == (BATCH, BLOCK) and tuple(by.shape) == (BATCH, BLOCK)

    ok = targets_are_next and stream_ok and splits_disjoint and shape_ok
    print("")
    print("  targets are the next token: %s" % ("YES" if (targets_are_next and stream_ok) else "NO"))
    print("  train and val splits disjoint: %s" % ("YES" if splits_disjoint else "NO"))
    print("  batches well shaped (%d x %d): %s" % (BATCH, BLOCK, "YES" if shape_ok else "NO"))
    print("")
    print("DATASET IS WELL-FORMED (targets are next tokens, splits disjoint): %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
