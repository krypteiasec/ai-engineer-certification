#!/usr/bin/env python3
"""
LAB TF4: The KV cache, why generation is fast.

When a model generates text it produces one token at a time. The naive way is to
re-run attention over the WHOLE sequence so far at every step. But the keys and
values of the earlier tokens never change once computed. Recomputing them is pure
waste. The KV cache stores each token's key and value the first time it is seen,
so every new step only projects the ONE new token and attends over the cached
rest. Same answer, far less work. This is why every serving stack (vLLM, TGI,
llama.cpp) is built around the KV cache.

You build both paths here with real torch tensors and PROVE two things:
  - the cached decode produces the EXACT same output as full recompute,
  - it does far fewer key/value projections (the FLOPs it saves).

Run: python3 modules/academy-content/labs/transformers/tf4-kv-cache.py
"""
import sys
import math
import torch
import torch.nn.functional as F

torch.manual_seed(13)  # reproducible

T, C = 6, 8            # generate T tokens, embedding width C
Wq = torch.randn(C, C)
Wk = torch.randn(C, C)
Wv = torch.randn(C, C)
stream = torch.randn(T, C)   # the token embeddings, revealed one at a time


def head_out(q_t, K, V):
    """Attention of a single query row q_t (C,) over cached K,V (t, C)."""
    scores = (K @ q_t) / math.sqrt(C)      # (t,)
    w = F.softmax(scores, dim=-1)          # causal by construction: only past+self
    return w @ V                           # (C,)


# PATH A: full recompute. At each step, re-project K and V for EVERY token 0..t.
full_out = []
full_proj = 0
for t in range(T):
    ctx = stream[:t + 1]                   # tokens seen so far
    K = ctx @ Wk                           # recompute all keys   (t+1, C)
    V = ctx @ Wv                           # recompute all values (t+1, C)
    full_proj += 2 * (t + 1)               # counted: 2 projections per past token
    q_t = stream[t] @ Wq
    full_out.append(head_out(q_t, K, V))
full_out = torch.stack(full_out)
print("STEP 1: full recompute done, %d tokens, %d key/value projections" % (T, full_proj))

# PATH B: KV cache. Project each token's K,V ONCE, append to the cache, reuse.
cache_K, cache_V = [], []
cached_out = []
cached_proj = 0
for t in range(T):
    k_t = stream[t] @ Wk                   # project ONLY the new token
    v_t = stream[t] @ Wv
    cached_proj += 2                       # counted: 2 projections total this step
    cache_K.append(k_t)
    cache_V.append(v_t)
    K = torch.stack(cache_K)               # (t+1, C) from cache, no recompute
    V = torch.stack(cache_V)
    q_t = stream[t] @ Wq
    cached_out.append(head_out(q_t, K, V))
cached_out = torch.stack(cached_out)
print("STEP 2: cached decode done, %d tokens, %d key/value projections" % (T, cached_proj))

# STEP 3: the outputs must match to floating point tolerance.
match = torch.allclose(full_out, cached_out, atol=1e-6)
max_diff = (full_out - cached_out).abs().max().item()
print("STEP 3: max output difference full-vs-cached = %.2e" % max_diff)

# STEP 4: the cache does strictly fewer projections. Full recompute is O(T^2) in
# projections (a triangular sum), the cache is O(T).
saved = full_proj - cached_proj
fewer = cached_proj < full_proj
print("STEP 4: projections  full=%d  cached=%d  saved=%d  (fewer FLOPs): %s" % (
    full_proj, cached_proj, saved, "YES" if fewer else "NO"))

ok = match and fewer
print("")
print("KV-CACHE DECODING MATCHES FULL RECOMPUTE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("Identical output, a fraction of the work. That is the trade the KV cache")
print("makes, and it is why token-by-token generation is affordable at all.")
