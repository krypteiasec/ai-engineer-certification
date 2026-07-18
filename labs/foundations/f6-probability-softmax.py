#!/usr/bin/env python3
"""
LAB F6: Probability and softmax. Turning scores into choices.

A model outputs raw scores. To pick a next token it needs PROBABILITIES: positive
numbers that sum to 1. The softmax function does exactly that, and it is used at
the output of every language model. You implement softmax and sampling here and
prove the output is a real probability distribution.

Run: python3 modules/academy-content/labs/foundations/f6-probability-softmax.py
"""
import sys, math

def softmax(scores):
    m = max(scores)                          # subtract the max for numerical safety
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def sample(probs, r):                        # r is a number in [0, 1)
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1

scores = [2.0, 1.0, 0.1, -1.0]
probs = softmax(scores)
print("STEP 1-2: softmax turns scores into probabilities")
for s, p in zip(scores, probs):
    print(f"  score {s:5.1f} -> prob {p:.3f}")

# a fixed, seedless pseudo-random walk so the sample is reproducible in the lab.
picks = [sample(probs, (i * 0.1) % 1.0) for i in range(10)]
print("")
print(f"STEP 3: sampling 10 times (reproducible): {picks}")

# STEP 4: the invariants. Probabilities are all >= 0 and sum to 1, and the highest
# score got the highest probability (softmax preserves order).
total = sum(probs)
ok = (abs(total - 1.0) < 1e-9) and all(p >= 0 for p in probs) and (probs.index(max(probs)) == scores.index(max(scores)))
print("")
print(f"STEP 4: probs sum to 1, all >= 0, and top score got top prob: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Softmax is the output of every LLM. Next: derivatives, the key to learning.")
