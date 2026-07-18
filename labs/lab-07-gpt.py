#!/usr/bin/env python3
"""
LAB 07: Assemble the GPT: stacking blocks into a model.

Now you wire the whole thing together into a real (tiny, untrained) GPT forward
pass, as one nn.Module. The path a sequence takes:
  tokens -> token embeddings + position embeddings
         -> N transformer blocks
         -> final layer norm
         -> lm_head: unembed to one score (logit) per vocabulary token
         -> softmax -> a probability for every possible next token.
With random weights the predictions are nonsense, but the SHAPE of the
computation is exactly GPT. Training (next chapter) is what makes the numbers
good. This is a faithful miniature of Karpathy's nanoGPT.

Run: python3 modules/academy-content/labs/lab-07-gpt.py
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(99)  # reproducible

VOCAB = list("abcdefg .")  # 9 toy tokens: a b c d e f g space period
V = len(VOCAB)
C = 8         # embedding width
N_LAYER = 3   # stack 3 transformer blocks
N_BLOCK = 6   # max sequence length the model can see (context window)
stoi = {c: i for i, c in enumerate(VOCAB)}


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(C, C, bias=False)
        self.query = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(N_BLOCK, N_BLOCK)))

    def forward(self, x):
        t = x.shape[0]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(C))
        scores = scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalAttention()
        self.ln2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)          # one row per token
        self.pos_emb = nn.Embedding(N_BLOCK, C)    # one row per position
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V)             # C-vector -> V scores

    def forward(self, ids):
        t = ids.shape[0]
        if t > N_BLOCK:
            raise ValueError("sequence longer than the context window")
        pos = torch.arange(t, dtype=torch.long)
        # STEP 1: embed tokens and ADD position embeddings (so order matters).
        x = self.tok_emb(ids) + self.pos_emb(pos)  # (T, C)
        # STEP 2: run through the stack of blocks.
        x = self.blocks(x)
        # STEP 3: final norm, then read out logits for EACH position.
        x = self.ln_f(x)
        return self.lm_head(x)  # (T, V)


model = TinyGPT()
seq = torch.tensor([stoi[c] for c in "bad ca"], dtype=torch.long)
print("STEP 1-4: forward pass on a %d-token sequence through %d blocks" % (seq.shape[0], N_LAYER))
logits = model(seq)                                  # (T, V)
# STEP 4: we predict the token after the LAST position -> softmax its logits.
probs = F.softmax(logits[-1], dim=-1)                # (V,)
print("  next-token probabilities (untrained, so roughly uniform):")
print("  " + "  ".join(
    "%s:%.3f" % ("_" if c == " " else c, probs[i].item()) for i, c in enumerate(VOCAB)))

# STEP 5: the invariant that proves it is a valid language model head: the output
# is a probability distribution over the WHOLE vocabulary (sums to 1), it has
# exactly V entries (one per possible next token), and every entry is >= 0.
valid = (abs(probs.sum().item() - 1.0) < 1e-6 and probs.shape[0] == V and bool((probs >= 0).all()))
print("")
print("STEP 5: output is a valid distribution over all %d tokens (sums to 1): %s" % (V, "YES" if valid else "NO"))
if not valid:
    sys.exit(1)
print("")
print("That is a complete GPT forward pass. The architecture is done. It just cannot")
print("predict well yet, because every weight is random. Next: TRAINING makes it learn.")
