#!/usr/bin/env python3
"""
LAB 05: Multi-head attention: many perspectives at once.

One attention head learns one way to relate tokens. Real models run several
heads in PARALLEL, each with its own query/key/value projections, so different
heads can specialize (one tracks the previous word, another tracks subjects, and
so on). You split the embedding into H slices, run a head on each, concatenate
the results, and project back to width C with an output linear. You build it as
real nn.Module objects (a Head and a MultiHead), the same shapes Karpathy uses.

Run: python3 modules/academy-content/labs/lab-05-multihead.py
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(11)  # reproducible

T, C, HEADS = 4, 8, 2
HEAD = C // HEADS  # each head gets C/HEADS dims -> 4


class Head(nn.Module):
    """One causal self-attention head (the Chapter 4 forward pass, as a module)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(T, T)))

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(q.shape[-1]))
        scores = scores.masked_fill(self.tril == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v  # (T, head_size)


class MultiHead(nn.Module):
    """HEADS heads run in parallel, concatenated, then projected back to width C."""

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(HEAD) for _ in range(HEADS)])
        self.proj = nn.Linear(C, C)  # mixes information across heads

    def forward(self, x):
        cat = torch.cat([h(x) for h in self.heads], dim=-1)  # (T, C)
        return self.proj(cat)


x = torch.randn(T, C)
mha = MultiHead()
print("STEP 1-2: input %dx%d, %d heads of size %d each" % (T, C, HEADS, HEAD))

# Show the concatenated (pre-projection) width, then the final projected output.
cat = torch.cat([h(x) for h in mha.heads], dim=-1)
out = mha(x)
print("STEP 3: concatenated heads -> %dx%d -> output projection -> %dx%d" % (
    cat.shape[0], cat.shape[1], out.shape[0], out.shape[1]))

# STEP 4: invariant: concatenating H heads of size C/H must reproduce width C,
# and the final output must still be T x C so blocks can stack.
width_ok = cat.shape[1] == C and out.shape[1] == C and out.shape[0] == T
print("STEP 4: width preserved through concat + projection (T x C): %s" % ("YES" if width_ok else "NO"))
if not width_ok:
    sys.exit(1)
print("")
print("Multiple heads, run in parallel, then merged. Next: wrap this in a full block.")
