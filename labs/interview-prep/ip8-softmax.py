#!/usr/bin/env python3
"""
LAB IP8: Softmax and numerically-stable softmax from scratch.

This is the most-asked from-scratch warmup in the question bank, and it hides a
trap. Anyone can write exp(x) / sum(exp(x)). The interviewer is waiting to see
if you know that the naive version OVERFLOWS: exp of a large logit becomes inf,
and inf / inf is nan. The fix is one line, subtract the max before exponenti
ating, and it is mathematically identical because softmax is shift-invariant.
Knowing WHY (shift invariance) is the point of the question.

You implement both and PROVE:
  - stable softmax matches a reference on normal inputs,
  - it hits the known value of softmax([1, 2, 3]),
  - the naive version overflows to nan on large logits while the stable version
    stays finite and still sums to 1.

Run: python3 modules/academy-content/labs/interview-prep/ip8-softmax.py
"""
import sys
import numpy as np


def naive_softmax(x):
    """The version people write first. Correct on small numbers, but exp() of a
    large value overflows to inf, and inf/inf is nan."""
    e = np.exp(x)
    return e / e.sum()


def stable_softmax(x):
    """Subtract the max first. Softmax is shift-invariant, so this returns the
    exact same distribution while keeping every exp() argument <= 0, which can
    never overflow."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


# 1. On normal inputs, stable softmax matches a trusted reference.
x = np.array([1.0, 2.0, 3.0])
ref = np.array([0.09003057, 0.24472847, 0.66524096])  # hand-computed reference
stable = stable_softmax(x)
naive = naive_softmax(x)
print("STEP 1: normal inputs")
print(f"  input        = {x.tolist()}")
print(f"  stable       = {np.round(stable, 8).tolist()}")
print(f"  reference    = {ref.tolist()}")
matches_ref = np.allclose(stable, ref, atol=1e-7)
agree_small = np.allclose(stable, naive, atol=1e-12)
sums_to_one = np.isclose(stable.sum(), 1.0)

# 2. On large logits the naive version blows up; the stable version does not.
big = np.array([1000.0, 1001.0, 1002.0])
naive_big = naive_softmax(big)
stable_big = stable_softmax(big)
print("")
print("STEP 2: large logits (overflow territory)")
print(f"  input        = {big.tolist()}")
print(f"  naive        = {naive_big.tolist()}  (nan means it overflowed)")
print(f"  stable       = {np.round(stable_big, 8).tolist()}")
naive_overflows = np.any(np.isnan(naive_big))
stable_finite = np.all(np.isfinite(stable_big)) and np.isclose(stable_big.sum(), 1.0)
# And shift invariance: softmax([1000,1001,1002]) == softmax([0,1,2]).
shift_invariant = np.allclose(stable_big, stable_softmax(np.array([0.0, 1.0, 2.0])), atol=1e-9)

print("")
print(f"        matches reference             : {matches_ref}")
print(f"        stable == naive on small nums : {agree_small}")
print(f"        distribution sums to 1        : {sums_to_one}")
print(f"        naive overflows to nan        : {naive_overflows}")
print(f"        stable stays finite and valid : {stable_finite}")
print(f"        shift invariance holds        : {shift_invariant}")

ok = bool(matches_ref and agree_small and sums_to_one and naive_overflows
          and stable_finite and shift_invariant)
print("")
print(f"STABLE SOFTMAX MATCHES REFERENCE AND HANDLES OVERFLOW: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("You now know the trap and the fix. That is the whole bank drill. Go get hired.")
