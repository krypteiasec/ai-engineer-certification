#!/usr/bin/env python3
"""
LAB TF8: Why transformers work, routing and order in one place.

Everything so far leads to one question: what makes this architecture win? Two
properties, and you prove both here with real tensors.

  1. DIRECT ROUTING. Attention connects any two positions in a SINGLE hop by
     content, not by distance. A recurrent net has to pass information step by
     step, so far-apart tokens are many hops apart and the signal fades. Attention
     builds an "induction head": find the earlier place this token appeared, and
     copy what came next. You hand-wire that head and watch it solve an exact
     copy/lookup task across an arbitrary gap, in one layer.

  2. ORDER NEEDS POSITIONS. Bare attention is permutation-equivariant: shuffle the
     tokens and the per-token answers just shuffle with them. That is WHY every
     transformer adds positional information (Lab TF1). You prove the order-blindness
     directly, which is the flip side of the same coin.

Run: python3 modules/academy-content/labs/transformers/tf8-why-transformers-work.py
"""
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(31)  # reproducible

# ---- Part 1: an induction head does long-range content-based lookup in one hop.
# Sequence of symbol ids. Task: for the LAST token, find its previous occurrence
# and predict the token that FOLLOWED it. Classic induction: [... A B ... A] -> B.
V = 12                       # vocabulary size
seq = [4, 7, 2, 9, 7, 5, 3, 8, 2, 9]   # note: '2' appears at idx 2 and idx 8
#                          the token after the FIRST '2' (idx 2) is '9'
query_pos = len(seq) - 1     # we ask: what follows the last token's earlier twin?
last = seq[-1]               # == 9

emb = torch.eye(V)           # one-hot embeddings: clean, exact content matching
X = emb[torch.tensor(seq)]   # (T, V)
T = X.shape[0]

# Hand-wire the induction head. Query/key = identity (match tokens by content),
# so token t attends to earlier positions holding the SAME symbol. The value at
# each position is the NEXT token, so the head copies "what came after the match".
def induction_predict(X, seq):
    T = X.shape[0]
    q = X[-1]                                  # content of the last token
    K = X[:-1]                                 # earlier tokens as keys
    scores = (K @ q) * 30.0                    # sharp match on identical content
    # value at position i = the token that FOLLOWED position i.
    nxt = torch.tensor([seq[i + 1] for i in range(T - 1)])
    Vval = emb[nxt]                            # (T-1, V)
    w = F.softmax(scores, dim=-1)
    return (w @ Vval).argmax().item()          # predicted next-token id

pred = induction_predict(X, seq)
# ground truth: find the earlier occurrence of `last`, take the token after it.
earlier = [i for i in range(T - 1) if seq[i] == last][0]
truth = seq[earlier + 1]
routed = pred == truth
print("STEP 1: induction head over %d tokens, gap = %d positions" % (T, query_pos - earlier))
print("        last token %d last appeared at idx %d, followed by %d" % (last, earlier, truth))
print("        head predicted %d (correct copy across the gap): %s" % (
    pred, "YES" if routed else "NO"))

# STEP 2: the routing is DIRECT. The match happens in one attention step no matter
# how far apart the two occurrences are. Widen the gap and it still works.
long_seq = [2, 9] + [1, 3, 5, 6, 8, 0, 11, 10, 4, 7] + [2]   # '2' at idx 0, gap 12
Xl = emb[torch.tensor(long_seq)]
pred_l = induction_predict(Xl, long_seq)
long_ok = pred_l == 9        # token after the first '2' is '9'
print("STEP 2: same head, gap widened to %d, still one hop, predicted %d: %s" % (
    len(long_seq) - 2, pred_l, "YES" if long_ok else "NO"))

# ---- Part 2: bare attention is order-blind, which is WHY positions are needed.
def bare_attention(X):
    """Full (non-causal) self-attention with identity projections, no positions."""
    scores = (X @ X.transpose(0, 1))
    W = F.softmax(scores, dim=-1)
    return W @ X

x = torch.randn(6, 8)
perm = torch.randperm(6)
out = bare_attention(x)
out_perm = bare_attention(x[perm])
# permutation-equivariant: attending over shuffled tokens = shuffle of the output.
equivariant = torch.allclose(out_perm, out[perm], atol=1e-6)
print("STEP 3: bare attention is permutation-equivariant (order-blind): %s" % (
    "YES" if equivariant else "NO"))
print("        => positional encodings are REQUIRED to give the model order")

ok = routed and long_ok and equivariant
print("")
print("TRANSFORMERS ROUTE ANY-DISTANCE INFO IN ONE HOP AND NEED POSITIONS FOR ORDER: %s" % (
    "YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("Direct content-based routing plus injected order. Those two facts, stacked")
print("and scaled, are the whole reason the transformer beat everything before it.")
