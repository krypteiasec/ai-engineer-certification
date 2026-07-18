#!/usr/bin/env python3
"""
LAB IP3: Scaled dot-product self-attention from scratch.

This is THE from-scratch ML coding ask. If an interviewer asks you to code one
thing at a whiteboard, it is attention. The whole operation is one line of math:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
but the round tests whether you can hold the tensor shapes straight, remember to
scale by sqrt(d_k), and (for a decoder) apply the causal mask so a token cannot
attend to the future.

You build it here with real torch tensors and PROVE:
  - the output shape is (seq_len, d_v), exactly what the next layer expects,
  - each attention row is a valid distribution that sums to 1,
  - the output matches an independent reference computation to 1e-6,
  - with a near-one-hot score, a query returns the value row it points at,
  - the causal mask zeroes all attention to future positions.

Run: python3 modules/academy-content/labs/interview-prep/ip3-attention.py
"""
import sys
import torch

torch.manual_seed(7)  # reproducible


def softmax_lastdim(x):
    """Numerically-stable softmax over the last dimension."""
    x = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)


def scaled_dot_product_attention(Q, K, V, causal=False):
    """Q: (seq, d_k), K: (seq, d_k), V: (seq, d_v). Returns (seq, d_v) plus the
    attention weight matrix (seq, seq) so we can inspect it."""
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)   # (seq, seq)
    if causal:
        seq = scores.shape[-1]
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()  # future = True
        scores = scores.masked_fill(mask, float("-inf"))
    weights = softmax_lastdim(scores)                   # (seq, seq)
    out = weights @ V                                   # (seq, d_v)
    return out, weights


# A small, fully specified case we can reason about by hand.
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
V = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

out, weights = scaled_dot_product_attention(Q, K, V)

# Independent reference: recompute the same math a different way, by hand-rolled
# numpy-free torch, and require the from-scratch output to match it.
d_k = Q.shape[-1]
ref_scores = (Q @ K.T) / (d_k ** 0.5)
ref_w = torch.softmax(ref_scores, dim=-1)   # torch's own softmax as the oracle
ref_out = ref_w @ V

print("STEP 1: shapes")
print(f"  Q,K: (seq, d_k) = {tuple(Q.shape)}   V: (seq, d_v) = {tuple(V.shape)}")
print(f"  output shape    = {tuple(out.shape)}  (must be (2, 2))")

print("")
print("STEP 2: attention weights (each row a distribution)")
print(f"  weights =\n{weights}")
row_sums = weights.sum(dim=-1)
print(f"  row sums = {row_sums.tolist()}  (must be 1.0)")

print("")
print("STEP 3: output vs independent reference")
print(f"  ours = {out.tolist()}")
print(f"  ref  = {ref_out.tolist()}")

# Known-value near-one-hot case: huge scale makes query 0 attend almost fully to
# key 0, so its output must equal V row 0.
Qs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
one_hot_out, one_hot_w = scaled_dot_product_attention(Qs, Qs, V)
picks_row0 = torch.allclose(one_hot_out[0], V[0], atol=1e-3)

# Causal mask: position 0 must place zero weight on the future position 1.
_, causal_w = scaled_dot_product_attention(Q, K, V, causal=True)
no_future_leak = torch.allclose(causal_w[0, 1], torch.tensor(0.0), atol=1e-9)

shape_ok = tuple(out.shape) == (2, 2)
sums_ok = torch.allclose(row_sums, torch.ones(2), atol=1e-6)
matches_ref = torch.allclose(out, ref_out, atol=1e-6)

print("")
print(f"        shape is (seq, d_v)          : {shape_ok}")
print(f"        rows sum to 1                : {sums_ok}")
print(f"        matches reference to 1e-6    : {matches_ref}")
print(f"        near-one-hot returns V row 0 : {picks_row0}")
print(f"        causal mask blocks the future: {no_future_leak}")

ok = shape_ok and sums_ok and matches_ref and picks_row0 and no_future_leak
print("")
print(f"SELF-ATTENTION OUTPUT MATCHES KNOWN CASE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("You coded the core of the transformer. Next round: theory with judgment.")
