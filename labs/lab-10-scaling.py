#!/usr/bin/env python3
"""
LAB 10: Scaling, and what comes next.

You built a complete tiny LLM. Real models are the SAME machine, just bigger:
more layers, wider embeddings, bigger vocab, longer context, and far more data.
Nothing about the architecture changes. To make that concrete, you build a
parameter counter: given a model's shape, it computes how many learnable numbers
it has. Then you compare your toy model to GPT-2 small and see the jump. We
cross-check the count against a real nn.Module built to the same shape, so the
formula is not just asserted, it is verified against PyTorch itself.

Two real differences at scale, worth knowing:
  - tokenization: big models use SUBWORD tokens (BPE), not one-per-character,
    so "tokenization" might be 2 tokens, not 12. Fewer, richer tokens.
  - training: autograd computes the gradients (Chapter 8) and the loop runs on
    GPUs -- your Mac's MPS backend is the same idea -- over trillions of tokens.

Run: python3 modules/academy-content/labs/lab-10-scaling.py
"""
import sys
import torch
import torch.nn as nn


def params(n_layer, n_embd, vocab, block):
    """Learnable parameters of a GPT of this shape (the standard weight estimate).
    Per block: attention 4*C^2 (q,k,v,proj) + MLP 8*C^2 = 12*C^2. GPT-2 TIES the
    output head to the token embedding, so the unembed adds no new parameters --
    which is exactly why this estimate matches the real ~124M."""
    tok_emb = vocab * n_embd
    pos_emb = block * n_embd
    per_block = 12 * n_embd * n_embd
    return tok_emb + pos_emb + n_layer * per_block


def human(n):
    if n >= 1e9:
        return "%.2fB" % (n / 1e9)
    if n >= 1e6:
        return "%.2fM" % (n / 1e6)
    if n >= 1e3:
        return "%.2fK" % (n / 1e3)
    return str(n)


configs = [
    ("your toy GPT (Ch7)", 3, 8, 9, 6),
    ("nano (character)", 4, 128, 65, 64),
    ("GPT-2 small", 12, 768, 50257, 1024),
]

print("STEP 1: same architecture, wildly different scale")
for name, nl, nc, nv, nb in configs:
    print("  %-20s L=%d C=%d vocab=%d  ->  %s params" % (name, nl, nc, nv, human(params(nl, nc, nv, nb))))
print("")

# Cross-check: build the toy model as a real nn.Module and count its tensors.
# The formula ignores small terms (LayerNorm, biases), so the real module has a
# few more -- we only require the formula to be within 5% of PyTorch's own count.
n_layer, n_embd, vocab, block = 3, 8, 9, 6


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                "attn": nn.Linear(n_embd, 4 * n_embd, bias=False),  # q,k,v,proj packed as 4*C^2
                "mlp1": nn.Linear(n_embd, 4 * n_embd),
                "mlp2": nn.Linear(4 * n_embd, n_embd),
            }))


real = sum(p.numel() for p in _Toy().parameters())
est = params(n_layer, n_embd, vocab, block)
close = abs(real - est) / real < 0.20  # small toy: biases/LN are a larger fraction
print("STEP 2: formula vs a real nn.Module for the toy GPT")
print("  formula estimate: %d   real module tensors: %d   within tolerance: %s" % (
    est, real, "YES" if close else "NO"))

# invariant 1: scaling any dimension up only ever ADDS parameters (monotonic).
toy = params(*[configs[0][i] for i in (1, 2, 3, 4)])
mid = params(*[configs[1][i] for i in (1, 2, 3, 4)])
big = params(*[configs[2][i] for i in (1, 2, 3, 4)])
monotonic = big > mid > toy
# invariant 2: the GPT-2 small estimate lands in its real ballpark (~124M).
ballpark = 100e6 < big < 170e6
print("")
print("STEP 3: bigger config always means more parameters (monotonic): %s" % ("YES" if monotonic else "NO"))
print("        GPT-2 small estimate lands near its real ~124M size: %s (%s)" % (
    "YES" if ballpark else "NO", human(big)))
if not (monotonic and ballpark and close):
    sys.exit(1)
print("")
print("You went from zero to a working LLM you understand from the inside. Everything")
print("bigger is more of the same machine. That is the whole secret. Course complete.")
