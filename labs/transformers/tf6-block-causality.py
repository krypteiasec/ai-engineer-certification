#!/usr/bin/env python3
"""
LAB TF6: The full block, and a hard proof of causality.

Course 1 built a transformer block and checked the attention weight matrix was
lower-triangular. That is good, but here you prove causality the way a scientist
would: by INTERVENTION. Assemble the whole block (pre-norm, causal attention,
residual, MLP, residual), then reach in and CHANGE a token at a future position.
If the block is truly causal, the outputs at every EARLIER position must not move
by a single bit, while the changed position and later ones do move. That
intervention test catches leaks a triangular-matrix check can miss (a stray bias,
a wrong transpose, a normalization that mixes across time).

You verify shape (T x C in, T x C out, so blocks stack) and run the intervention.

Run: python3 modules/academy-content/labs/transformers/tf6-block-causality.py
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(23)  # reproducible

T, C = 6, 8


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(C, C, bias=False)
        self.k = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(T, T)))

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(C)
        scores = scores.masked_fill(self.tril == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-norm transformer block: the exact unit every GPT stacks."""

    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.attn, self.mlp = CausalAttention(), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # tokens share information (causally)
        x = x + self.mlp(self.ln2(x))    # each token thinks on its own
        return x


block = Block().eval()
x = torch.randn(T, C)

# STEP 1: shape in == shape out, so blocks stack into a deep model.
with torch.no_grad():
    out = block(x)
shape_ok = out.shape == (T, C)
print("STEP 1: block maps %s -> %s (stackable): %s" % (
    tuple(x.shape), tuple(out.shape), "YES" if shape_ok else "NO"))

# STEP 2: intervention. Perturb the token at a FUTURE position j and re-run.
j = 3
x2 = x.clone()
x2[j] = torch.randn(C)          # completely change the token at position j
with torch.no_grad():
    out2 = block(x2)

# STEP 3: measure how much each position's output moved.
delta = (out - out2).abs().max(dim=-1).values      # (T,) per-position max change
print("STEP 3: perturbed token at position %d, per-position output change:" % j)
for i, d in enumerate(delta.tolist()):
    tag = "past  " if i < j else ("CHANGED" if i == j else "future")
    print("   t%d %s change=%.3e" % (i, tag, d))

# STEP 4: the invariant. Positions BEFORE j must be untouched (a token cannot see
# the future); position j and after must react.
past_frozen = delta[:j].abs().max().item() < 1e-6
future_reacts = delta[j:].abs().max().item() > 1e-4
print("STEP 4: earlier tokens unchanged: %s | changed+later tokens reacted: %s" % (
    "YES" if past_frozen else "NO", "YES" if future_reacts else "NO"))

ok = shape_ok and past_frozen and future_reacts
print("")
print("CAUSAL MASK BLOCKS FUTURE TOKENS: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("Editing the future left the past bit-for-bit identical. That is causality,")
print("proven by intervention, and it is what lets a decoder predict left to right.")
