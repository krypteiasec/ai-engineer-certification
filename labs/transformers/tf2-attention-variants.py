#!/usr/bin/env python3
"""
LAB TF2: Attention variants, MHA vs MQA vs GQA.

Multi-head attention (MHA) gives every query head its OWN key and value head. At
inference the KV cache stores those keys and values for every past token, and
with many heads that cache becomes the memory bottleneck. Two fixes shrink it:
  - MQA (multi-query): all query heads SHARE one single key/value head.
  - GQA (grouped-query): query heads share key/value heads in G groups. It is the
    dial between the two ends. Llama 2 70B, Llama 3, and Mistral all use GQA.

The beautiful part: GQA is a strict generalization. Set G = H (one KV head per
query head) and GQA IS ordinary MHA. Set G = 1 and GQA IS MQA. You build all
three from scratch here, share the weights, and PROVE those two identities hold
bit-for-bit, then show the KV cache shrinks as you use fewer KV heads.

Run: python3 modules/academy-content/labs/transformers/tf2-attention-variants.py
"""
import sys
import math
import torch
import torch.nn.functional as F

torch.manual_seed(5)  # reproducible

T, C, H = 5, 8, 4          # tokens, embedding width, query heads
HD = C // H                # head dim = 2
x = torch.randn(T, C)
tril = torch.tril(torch.ones(T, T))


def attn(q, k, v):
    """Causal scaled-dot-product attention for one head. q,k,v: (T, HD)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(tril == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


# Shared projection weights so the three variants are comparable.
Wq = torch.randn(C, C)              # query: always H full heads -> C
Wk_full = torch.randn(C, C)        # MHA keys: H heads -> C
Wv_full = torch.randn(C, C)        # MHA values: H heads -> C
Wk_one = torch.randn(C, HD)        # MQA keys: 1 head
Wv_one = torch.randn(C, HD)        # MQA values: 1 head


def split(t, n):
    """Split width into n heads of HD along the last dim -> list of (T, HD)."""
    return [t[:, h * HD:(h + 1) * HD] for h in range(n)]


def mha():
    """H query heads, H KV heads. Standard multi-head attention."""
    q, k, v = split(x @ Wq, H), split(x @ Wk_full, H), split(x @ Wv_full, H)
    return torch.cat([attn(q[h], k[h], v[h]) for h in range(H)], dim=-1)


def mqa():
    """H query heads, ONE shared KV head."""
    q = split(x @ Wq, H)
    k1, v1 = x @ Wk_one, x @ Wv_one
    return torch.cat([attn(q[h], k1, v1) for h in range(H)], dim=-1)


def gqa(G, Wk, Wv):
    """H query heads sharing KV in G groups. Wk,Wv project to G*HD."""
    q = split(x @ Wq, H)
    k, v = split(x @ Wk, G), split(x @ Wv, G)
    per_group = H // G
    return torch.cat([attn(q[h], k[h // per_group], v[h // per_group]) for h in range(H)], dim=-1)


# STEP 1: GQA with G = H must equal MHA exactly (same weights, full-width KV).
g_is_mha = gqa(H, Wk_full, Wv_full)
mha_out = mha()
id_mha = torch.allclose(g_is_mha, mha_out, atol=1e-6)
print("STEP 1: GQA(G=H) == MHA (bit-for-bit): %s" % ("YES" if id_mha else "NO"))

# STEP 2: GQA with G = 1 must equal MQA exactly (one shared KV head).
g_is_mqa = gqa(1, Wk_one, Wv_one)
mqa_out = mqa()
id_mqa = torch.allclose(g_is_mqa, mqa_out, atol=1e-6)
print("STEP 2: GQA(G=1) == MQA (bit-for-bit): %s" % ("YES" if id_mqa else "NO"))

# STEP 3: a middle GQA (G=2) still produces the same (T, C) output shape, so it
# drops into any model unchanged.
Wk_mid, Wv_mid = torch.randn(C, 2 * HD), torch.randn(C, 2 * HD)
mid = gqa(2, Wk_mid, Wv_mid)
shape_ok = mid.shape == (T, C)
print("STEP 3: GQA(G=2) output shape %s (want (%d, %d)): %s" % (
    tuple(mid.shape), T, C, "YES" if shape_ok else "NO"))

# STEP 4: KV cache size scales with the number of KV heads. Per token we cache
# one key and one value per KV head: 2 * G * HD floats. Fewer KV heads => smaller
# cache. This is the entire reason MQA/GQA exist.
def kv_cache_floats(G):
    return 2 * G * HD * T   # keys + values, G heads, HD each, T tokens

cache_mha = kv_cache_floats(H)   # G = H
cache_gqa = kv_cache_floats(2)
cache_mqa = kv_cache_floats(1)
shrinks = cache_mha > cache_gqa > cache_mqa
print("STEP 4: KV cache floats  MHA=%d  GQA(G=2)=%d  MQA=%d  (monotonic shrink): %s" % (
    cache_mha, cache_gqa, cache_mqa, "YES" if shrinks else "NO"))
print("        MQA cache is %dx smaller than MHA for the same output shape" % (cache_mha // cache_mqa))

ok = id_mha and id_mqa and shape_ok and shrinks
print("")
print("GQA UNIFIES MHA (G=H) AND MQA (G=1), FEWER KV HEADS SHRINK THE CACHE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("One knob, G, trades KV memory against capacity. That is the whole design.")
