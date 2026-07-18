#!/usr/bin/env python3
"""
LAB 06: The transformer block: residuals, layer norm, and an MLP.

A transformer block is the unit you stack to make a model deep. It has two
sub-layers: attention (tokens share information) and an MLP (each token thinks on
its own). Around each, two tricks make deep networks trainable:
  - layer norm: rescale a vector to mean 0, variance 1, so numbers stay tame,
  - residual connection: add the input back to the output (x + sublayer(x)), so
    gradients have a clean path and the block can learn a small change.
You build the block as a real nn.Module using nn.LayerNorm, nn.Linear, and
F.gelu. It takes T x C in and returns T x C out, which is exactly why blocks
stack.

Run: python3 modules/academy-content/labs/lab-06-block.py
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(21)  # reproducible

T, C = 4, 8


class CausalAttention(nn.Module):
    """One causal self-attention head at full width C (Chapter 4 forward)."""

    def __init__(self):
        super().__init__()
        self.key = nn.Linear(C, C, bias=False)
        self.query = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(T, T)))

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(C))
        scores = scores.masked_fill(self.tril == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class MLP(nn.Module):
    """Expand to 4x width, apply GELU, project back. Each token, independently."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-norm transformer block: norm before each sublayer, with residuals."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalAttention()
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # x + attention(norm(x))
        x = x + self.mlp(self.ln2(x))   # x + mlp(norm(x))
        return x


x = torch.randn(T, C)
block = Block()
print("STEP 1-3: input %dx%d -> transformer block" % (T, C))
out = block(x)
print("  block output: %dx%d" % (out.shape[0], out.shape[1]))

# invariant 1: shape in == shape out (so blocks stack).
shape_ok = out.shape[0] == T and out.shape[1] == C
# invariant 2: layer norm really produces ~mean 0 per token row.
normed = nn.LayerNorm(C)(x)
mean_ok = bool(normed.mean(dim=-1).abs().max().item() < 1e-5)
print("")
print("STEP 4: shape preserved (T x C, stackable): %s" % ("YES" if shape_ok else "NO"))
print("        layer norm gives mean ~0 per token: %s" % ("YES" if mean_ok else "NO"))
if not (shape_ok and mean_ok):
    sys.exit(1)
print("")
print("This block is the LEGO brick of every GPT. Next: stack several into a model.")
