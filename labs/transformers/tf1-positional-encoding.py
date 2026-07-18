#!/usr/bin/env python3
"""
LAB TF1: Positional encodings, from sinusoidal to RoPE.

Attention has no built-in sense of order. Scramble the tokens and, without
positions, the math gives the same answer. So every transformer injects position
somehow. The old way (the original 2017 paper) ADDS a fixed sinusoidal vector to
each token: absolute position baked into the embedding. The modern way (RoPE,
used by Llama, Mistral, Qwen) ROTATES the query and key vectors by an angle
proportional to their position. The payoff is a clean mathematical property: the
attention score between a query at position m and a key at position n depends
only on the RELATIVE offset (m - n), not on where the pair sits in the sequence.

You build both here with real torch tensors and PROVE the difference:
  - sinusoidal-add: the q.k score for a fixed offset drifts as you slide the pair
    down the sequence (it leaks absolute position),
  - RoPE: the q.k score for a fixed offset is invariant no matter where the pair
    sits.

Run: python3 modules/academy-content/labs/transformers/tf1-positional-encoding.py
"""
import sys
import math
import torch

torch.manual_seed(7)  # reproducible

D = 8            # head dimension (even, so it splits into D/2 rotated pairs)
BASE = 10000.0   # the standard RoPE / sinusoidal frequency base


def inv_freqs(d, base):
    """theta_i = base^(-2i/d) for i in 0..d/2-1. Low dims spin fast, high dims slow."""
    i = torch.arange(0, d, 2, dtype=torch.float32)  # 0,2,4,...
    return base ** (-i / d)                          # (d/2,)


def sinusoidal(pos, d, base):
    """The 2017 fixed positional vector for one absolute position `pos`."""
    theta = inv_freqs(d, base)          # (d/2,)
    ang = pos * theta                   # (d/2,)
    pe = torch.zeros(d)
    pe[0::2] = torch.sin(ang)
    pe[1::2] = torch.cos(ang)
    return pe                           # (d,)


def rope(x, pos, base):
    """Rotate each (x[2i], x[2i+1]) pair by angle pos*theta_i. This IS RoPE."""
    theta = inv_freqs(x.shape[-1], base)   # (d/2,)
    ang = pos * theta                      # (d/2,)
    cos, sin = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[0::2], x[1::2]               # even and odd coords of each pair
    out = torch.empty_like(x)
    out[0::2] = x0 * cos - x1 * sin        # standard 2D rotation
    out[1::2] = x0 * sin + x1 * cos
    return out


# One fixed query and key vector. We will place them at many positions.
q = torch.randn(D)
k = torch.randn(D)
print("STEP 1: fixed query and key, dim %d (%d rotated pairs)" % (D, D // 2))

# STEP 2: sinusoidal ADD. score = (q + PE_m) . (k + PE_n). Hold the offset fixed
# at OFFSET and slide the pair down the sequence; watch the score drift.
OFFSET = 2
starts = [2, 5, 9, 14, 20]
print("STEP 2: sinusoidal-add scores at fixed offset %d, sliding the pair:" % OFFSET)
sin_scores = []
for m in starts:
    n = m - OFFSET
    s = torch.dot(q + sinusoidal(m, D, BASE), k + sinusoidal(n, D, BASE)).item()
    sin_scores.append(s)
    print("   m=%2d n=%2d  score=% .4f" % (m, n, s))
sin_spread = max(sin_scores) - min(sin_scores)

# STEP 3: RoPE. score = rope(q, m) . rope(k, n). Same fixed offset, sliding pair.
print("STEP 3: RoPE scores at the SAME fixed offset %d, sliding the pair:" % OFFSET)
rope_scores = []
for m in starts:
    n = m - OFFSET
    s = torch.dot(rope(q, m, BASE), rope(k, n, BASE)).item()
    rope_scores.append(s)
    print("   m=%2d n=%2d  score=% .4f" % (m, n, s))
rope_spread = max(rope_scores) - min(rope_scores)

# STEP 4: different offsets DO give different RoPE scores (positions still matter,
# just relatively). Prove the score is a real function of the offset, not a
# constant that trivially "does not move".
print("STEP 4: RoPE score genuinely depends on the offset (not a constant):")
by_offset = []
for off in [0, 1, 3, 6]:
    m = 12
    s = torch.dot(rope(q, m, BASE), rope(k, m - off, BASE)).item()
    by_offset.append(s)
    print("   offset=%d  score=% .4f" % (off, s))
offsets_differ = (max(by_offset) - min(by_offset)) > 1e-3

# STEP 5: the invariants.
#  (a) RoPE is relative: same offset => same score no matter the absolute pos.
#  (b) sinusoidal-add is NOT relative: same offset drifts with absolute pos.
rope_invariant = rope_spread < 1e-5
sin_leaks = sin_spread > 1e-3
print("")
print("STEP 5: RoPE score spread at fixed offset = %.2e (want ~0)" % rope_spread)
print("        sinusoidal-add spread at fixed offset = %.4f (leaks absolute pos)" % sin_spread)

ok = rope_invariant and sin_leaks and offsets_differ
print("")
print("ROPE IS RELATIVE-POSITION INVARIANT: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("Same relative distance, same attention score, anywhere in the sequence.")
print("That single property is why modern models rotate instead of add.")
