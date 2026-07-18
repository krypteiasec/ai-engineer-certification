#!/usr/bin/env python3
"""
LAB 04: Self-attention: the idea that changed everything.

A bigram only sees the single previous character. Attention lets every position
look back at ALL earlier positions and decide, on its own, which ones matter.
Each token emits a query (what am I looking for), a key (what do I offer), and a
value (what I will pass on if chosen). We score every query against every key,
mask out the future (a token cannot see ahead), softmax the scores into weights,
and take a weighted sum of the values. That is one head of self-attention, and
you build it here with real torch tensors and nn.Linear projections: this is
literally Karpathy's `Head` module.

Run: python3 modules/academy-content/labs/lab-04-attention.py
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(3)  # reproducible

T = 4      # sequence length (4 tokens)
C = 6      # embedding size per token
HEAD = 6   # head size (== C here for one full head)

# STEP 1: a toy input: T tokens, each a C-dim embedding (as if from Chapter 2).
x = torch.randn(T, C)
print("STEP 1: input x is %d tokens x %d dims" % (T, C))

# STEP 2: three learnable projections turn each token into a query, key, value.
# nn.Linear(..., bias=False) is exactly what a transformer uses for q/k/v.
key = nn.Linear(C, HEAD, bias=False)
query = nn.Linear(C, HEAD, bias=False)
value = nn.Linear(C, HEAD, bias=False)
q = query(x)  # (T, HEAD)
k = key(x)    # (T, HEAD)
v = value(x)  # (T, HEAD)
print("STEP 2: projected x into query, key, value (each %dx%d)" % (T, HEAD))

# STEP 3: attention scores = q . k^T, scaled by 1/sqrt(head) to keep them tame.
scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(HEAD))  # (T, T)

# STEP 4: causal mask. Token i may only attend to tokens j <= i (no peeking
# ahead). torch.tril gives the lower triangle; we set the future to -inf so
# softmax gives it zero weight. This is the masked_fill idiom every GPT uses.
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float("-inf"))

# STEP 5: softmax each row -> attention weights that sum to 1.
weights = F.softmax(scores, dim=-1)  # (T, T)
print("")
print("STEP 3-5: attention weights (row i = how much token i attends to each earlier token)")
for i, row in enumerate(weights.tolist()):
    print("  t%d: [%s]" % (i, ", ".join("%.2f" % w for w in row)))

# STEP 6: output = weighted sum of values. Each token becomes a blend of the
# values it chose to attend to.
out = weights @ v  # (T, HEAD)
print("")
print("STEP 6: output is %dx%d (a context-aware vector per token)" % (out.shape[0], out.shape[1]))

# STEP 7: the invariants that prove attention is correct:
#  (a) every weight row sums to 1 (it is a distribution),
#  (b) the mask holds: token i puts ZERO weight on any future token j > i.
sums_ok = bool(torch.allclose(weights.sum(dim=-1), torch.ones(T), atol=1e-6))
upper = weights * (mask == 0)          # everything strictly above the diagonal
causal_ok = bool(upper.abs().max().item() < 1e-9)
print("")
print("STEP 7: weight rows sum to 1: %s | causal (no peeking ahead): %s" % (
    "YES" if sums_ok else "NO", "YES" if causal_ok else "NO"))
if not (sums_ok and causal_ok):
    sys.exit(1)
print("")
print("You just computed self-attention with real torch tensors. Next: run several heads.")
