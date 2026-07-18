#!/usr/bin/env python3
"""
LAB IP7: Temperature, top-k, and top-p sampling from scratch.

Owning a probabilistic system, the theme every behavioral round probes, starts
with understanding the exact knobs that make an LLM nondeterministic. Sampling
is where "the same prompt gave a different answer" comes from, and a senior
engineer can explain and control it. Three levers, all applied to the logits
before you draw a token:
  - temperature: divide the logits by T. T -> 0 sharpens toward greedy (argmax);
    T > 1 flattens the distribution toward random.
  - top-k: keep only the k highest-probability tokens, renormalize, sample.
  - top-p (nucleus): keep the smallest set of tokens whose cumulative
    probability first reaches p, renormalize, sample.

You implement all three and PROVE their defining distribution properties.

Run: python3 modules/academy-content/labs/interview-prep/ip7-sampling.py
"""
import sys
import numpy as np


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits, t):
    """Scale logits by 1/T. Small T sharpens, large T flattens. T must be > 0."""
    return logits / t


def top_k_filter(probs, k):
    """Zero out everything except the k highest-probability tokens, renormalize."""
    if k >= len(probs):
        return probs.copy()
    keep = np.argsort(probs)[-k:]           # indices of the top k
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum()


def top_p_filter(probs, p):
    """Keep the smallest set of tokens whose cumulative probability reaches p,
    renormalize. This is nucleus sampling."""
    order = np.argsort(probs)[::-1]         # high to low
    cumulative = np.cumsum(probs[order])
    # Number of tokens needed to first reach p.
    cutoff = np.searchsorted(cumulative, p) + 1
    keep = order[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum()


def entropy(probs):
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log(nz)))


logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
base = softmax(logits)

# 1. Temperature: cold sharpens toward the top token, hot flattens.
cold = softmax(apply_temperature(logits, 0.1))
hot = softmax(apply_temperature(logits, 5.0))
print("STEP 1: temperature")
print(f"  base top prob = {base.max():.4f}")
print(f"  cold (T=0.1) top prob = {cold.max():.4f}  entropy = {entropy(cold):.4f}")
print(f"  hot  (T=5.0) top prob = {hot.max():.4f}  entropy = {entropy(hot):.4f}")
temp_ok = cold.max() > base.max() > hot.max() and entropy(hot) > entropy(base) > entropy(cold)

# 2. top-k = 1 puts all mass on the single best token.
k1 = top_k_filter(base, 1)
print("")
print("STEP 2: top-k")
print(f"  top-k=1 distribution = {np.round(k1, 4).tolist()}")
topk_ok = np.isclose(k1.max(), 1.0) and np.isclose(k1.sum(), 1.0) and np.argmax(k1) == np.argmax(base)

# 3. top-p keeps the nucleus and renormalizes to a valid distribution.
p90 = top_p_filter(base, 0.9)
kept = int(np.sum(p90 > 0))
print("")
print("STEP 3: top-p (nucleus)")
print(f"  top-p=0.9 keeps {kept} of {len(base)} tokens, sum = {p90.sum():.4f}")
# The nucleus must be a valid distribution, smaller than the full vocab, and it
# must include the most probable token.
topp_ok = np.isclose(p90.sum(), 1.0) and kept < len(base) and p90[np.argmax(base)] > 0

# 4. Determinism: greedy sampling (argmax) with a fixed seed always returns the
#    same token, the property you rely on to make a system reproducible.
rng = np.random.default_rng(42)
draws = [int(rng.choice(len(k1), p=k1)) for _ in range(1000)]
greedy_ok = all(d == np.argmax(base) for d in draws)
print("")
print(f"        temperature sharpens/flattens : {temp_ok}")
print(f"        top-k=1 is one-hot            : {topk_ok}")
print(f"        top-p nucleus is valid        : {topp_ok}")
print(f"        greedy sampling is stable     : {greedy_ok}")

ok = bool(temp_ok and topk_ok and topp_ok and greedy_ok)
print("")
print(f"SAMPLING KNOBS OBEY THEIR DISTRIBUTION PROPERTIES: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("You can now explain nondeterminism and control it. Next round: the bank.")
