#!/usr/bin/env python3
"""
Shared TinyGPT architecture for the Academy trained-model artifacts.

This is the EXACT same model students build in Course 1 (labs 07/08/10):
  tokens -> token embeddings + position embeddings
         -> N causal transformer blocks (single-head attention + MLP)
         -> final layer norm
         -> lm_head unembed to one logit per vocabulary token.

The only difference from lab-07's hardcoded toy is that the dimensions
(vocab, width, layers, context) are passed in as a config, so the SAME
machine can be scaled up enough to actually learn from a real corpus.
That is the literal lesson of lab-10: same architecture, bigger numbers.

LoRA (low-rank adapters) live here too so the fine-tune reuses this exact
model rather than a second copy. Nothing here needs a network.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """Single-head causal self-attention, identical in spirit to lab-07.

    Works on batched (B, T, C) or unbatched (T, C) tensors. The lm attends
    only to the present and past (the tril mask), never the future.
    """

    def __init__(self, c, n_block):
        super().__init__()
        self.c = c
        self.key = nn.Linear(c, c, bias=False)
        self.query = nn.Linear(c, c, bias=False)
        self.value = nn.Linear(c, c, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(n_block, n_block)))

    def forward(self, x):
        t = x.shape[-2]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.c))
        scores = scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)
        return self.proj(att @ v)


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, both residual.

    Byte-for-byte the lab-07 Block, just with the width passed in.
    """

    def __init__(self, c, n_block):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.attn = CausalAttention(c, n_block)
        self.ln2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """The Course-1 GPT, made config-driven so it can be trained for real."""

    def __init__(self, vocab, c, n_layer, n_block):
        super().__init__()
        self.vocab = vocab
        self.c = c
        self.n_layer = n_layer
        self.n_block = n_block
        self.tok_emb = nn.Embedding(vocab, c)
        self.pos_emb = nn.Embedding(n_block, c)
        self.blocks = nn.Sequential(*[Block(c, n_block) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(c)
        self.lm_head = nn.Linear(c, vocab)

    def forward(self, ids):
        # accept (T,) or (B, T); normalise to batched then restore.
        squeeze = ids.dim() == 1
        if squeeze:
            ids = ids.unsqueeze(0)
        t = ids.shape[1]
        if t > self.n_block:
            raise ValueError("sequence longer than the context window")
        pos = torch.arange(t, dtype=torch.long, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        return logits.squeeze(0) if squeeze else logits


# ----------------------------------------------------------------------------
# LoRA: freeze the base, learn only a low-rank correction on chosen weights.
# ----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update B @ A.

    output = base(x) + scaling * (x @ A^T @ B^T)
    Only A and B carry gradients; the base weight is frozen. This is the real
    LoRA formulation (Hu et al. 2021), just at toy scale.
    """

    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))  # starts at 0 => no-op at init

    def forward(self, x):
        base = self.base(x)
        lora = (x @ self.A.t()) @ self.B.t()
        return base + self.scaling * lora


def inject_lora(model, r=8, alpha=16):
    """Replace the query and value projections in every block with LoRA-wrapped
    versions. Returns the list of trainable LoRA parameters. The base model is
    frozen first, so training touches only the adapters."""
    for p in model.parameters():
        p.requires_grad = False
    lora_params = []
    for blk in model.blocks:
        blk.attn.query = LoRALinear(blk.attn.query, r=r, alpha=alpha)
        blk.attn.value = LoRALinear(blk.attn.value, r=r, alpha=alpha)
        for m in (blk.attn.query, blk.attn.value):
            lora_params.append(m.A)
            lora_params.append(m.B)
    return lora_params


def lora_state_dict(model):
    """Just the adapter tensors, keyed by block index and projection."""
    sd = {}
    for i, blk in enumerate(model.blocks):
        for name in ("query", "value"):
            mod = getattr(blk.attn, name)
            if isinstance(mod, LoRALinear):
                sd["blocks.%d.attn.%s.A" % (i, name)] = mod.A.detach().cpu()
                sd["blocks.%d.attn.%s.B" % (i, name)] = mod.B.detach().cpu()
    return sd


def load_lora_state_dict(model, sd):
    """Apply saved adapter tensors onto a LoRA-injected model."""
    for i, blk in enumerate(model.blocks):
        for name in ("query", "value"):
            mod = getattr(blk.attn, name)
            if isinstance(mod, LoRALinear):
                mod.A.data.copy_(sd["blocks.%d.attn.%s.A" % (i, name)])
                mod.B.data.copy_(sd["blocks.%d.attn.%s.B" % (i, name)])


# ----------------------------------------------------------------------------
# Tokenizer (char level) and generation.
# ----------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, vocab_chars):
        self.chars = list(vocab_chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


@torch.no_grad()
def generate(model, tok, prompt, n=200, temperature=0.0, device="cpu"):
    """Greedy (temperature 0) or sampled continuation. Temperature 0 is fully
    deterministic: it always takes the argmax token, so the same prompt yields
    the same text every run. That determinism is what verify.py checks."""
    model.eval()
    ids = tok.encode(prompt)
    if not ids:
        ids = [0]
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    for _ in range(n):
        ctx = ids[-model.n_block:]
        logits = model(ctx)
        last = logits[-1]
        if temperature <= 0.0:
            nxt = int(torch.argmax(last))
        else:
            probs = F.softmax(last / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        ids = torch.cat([ids, torch.tensor([nxt], dtype=torch.long, device=device)])
    return tok.decode(ids.tolist())
