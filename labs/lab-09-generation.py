#!/usr/bin/env python3
"""
LAB 09: Generation: sampling text from your model.

You trained a model in Chapter 8. Now you make it TALK. Generation is a loop:
feed the current token, get the logits over the next token, SAMPLE one from them,
append it, repeat. Two knobs control the flavor:
  - temperature: divides the logits. Below 1 makes the model confident and
    repetitive, above 1 makes it wild and random.
  - top-k: only sample from the k most likely tokens (torch.topk), cutting off
    the long tail of nonsense.
You retrain the same tiny bigram from Chapter 8 here (each lab runs on its own),
then sample from it with the real primitives: F.softmax and torch.multinomial.

Run: python3 modules/academy-content/labs/lab-09-generation.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = "the cat ran. the dog ran. the cat sat. a dog sat. the rat sat."
vocab = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(vocab)}
V = len(vocab)


def train(steps=200, lr=0.5):
    """The Chapter 8 loop, retrained here so this lab stands alone."""
    torch.manual_seed(1234)
    xs = torch.tensor([stoi[CORPUS[i]] for i in range(len(CORPUS) - 1)], dtype=torch.long)
    ys = torch.tensor([stoi[CORPUS[i + 1]] for i in range(len(CORPUS) - 1)], dtype=torch.long)
    W = nn.Embedding(V, V)
    opt = torch.optim.AdamW(W.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(W(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W


W = train()


def next_probs(logits, temperature, topk):
    """Softmax with temperature + optional top-k truncation."""
    logits = logits / max(temperature, 1e-6)
    if 0 < topk < V:
        vals, _ = torch.topk(logits, topk)         # the k largest logits
        cutoff = vals[-1]                          # smallest kept logit
        logits = torch.where(logits >= cutoff, logits, torch.tensor(float("-inf")))
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def generate(start, n, temperature, topk, seed):
    torch.manual_seed(seed)
    cur = stoi[start]
    out = [start]
    for _ in range(n):
        logits = W(torch.tensor(cur, dtype=torch.long))  # row for the current token
        probs = next_probs(logits, temperature, topk)
        nxt = int(torch.multinomial(probs, num_samples=1).item())
        out.append(vocab[nxt])
        cur = nxt
    return "".join(out)


print("STEP 1-2: generate from the model trained in Chapter 8")
print('  temperature 1.0 (balanced):  "%s"' % generate("t", 30, 1.0, 0, 5))
print('  temperature 0.4 (confident): "%s"' % generate("t", 30, 0.4, 0, 5))
print('  temperature 1.0, top-k 3:    "%s"' % generate("t", 30, 1.0, 3, 5))
print("")

# invariant: every generated character is a real vocabulary token, and the
# output has exactly the requested length (start + n sampled tokens).
g = generate("t", 30, 1.0, 0, 5)
valid_chars = all(c in stoi for c in g)
right_len = len(g) == 31
print("STEP 3: output uses only real vocab tokens: %s | correct length: %s" % (
    "YES" if valid_chars else "NO", "YES" if right_len else "NO"))
if not (valid_chars and right_len):
    sys.exit(1)
print("")
print("Your tiny model, trained by you, generating text. This is the full loop of an")
print("LLM: tokenize, embed, attend, predict, train, sample. You built every piece.")
