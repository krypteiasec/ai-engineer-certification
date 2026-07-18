#!/usr/bin/env python3
"""
LAB 03: The bigram model: your first real predictor.

A bigram model answers one question: given the current character, what is the
next one likely to be? You learn it by COUNTING. Walk the text, tally every pair
(current -> next) into a matrix, normalize the tallies into probabilities, and
you have a real generative model. It is simple and a little dumb, but it
genuinely learns the statistics of the data and can generate new text. This is
the exact starting point Karpathy uses in makemore, and we do it the makemore
way: a count TENSOR and torch.multinomial for sampling.

Run: python3 modules/academy-content/labs/lab-03-bigram.py
"""
import sys
import torch

torch.manual_seed(7)  # reproducible sampling

CORPUS = "the cat ran. the dog ran. the cat sat. a dog sat."

# STEP 1: vocabulary (chars). The "." boundary token is already in the text.
vocab = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(vocab)}
V = len(vocab)

# STEP 2: count every adjacent pair into a V x V tensor. counts[a, b] = how
# often b follows a. This IS the model: one count matrix.
counts = torch.zeros((V, V), dtype=torch.float)
for i in range(len(CORPUS) - 1):
    a, b = stoi[CORPUS[i]], stoi[CORPUS[i + 1]]
    counts[a, b] += 1
print("STEP 1-2: learned a %dx%d count tensor from the text" % (V, V))

# STEP 3: normalize each row into a probability distribution over the next char.
# Add 1 to every count first (Laplace smoothing) so nothing is impossible, then
# divide each row by its sum. keepdim=True keeps the broadcast shape correct.
probs = counts + 1.0
probs = probs / probs.sum(dim=1, keepdim=True)

# STEP 4: sanity check the probabilities: every row must sum to 1 (it IS a
# distribution). If a row does not sum to 1, sampling is meaningless.
row_sums = probs.sum(dim=1)
rows_ok = bool(torch.allclose(row_sums, torch.ones(V), atol=1e-6))
print("STEP 3-4: every row is a valid probability distribution (sums to 1): %s" % ("YES" if rows_ok else "NO"))
if not rows_ok:
    sys.exit(1)


# STEP 5: sample. Draw the next char from the current char's row distribution.
# torch.multinomial is the real sampling primitive every LLM uses at generation.
def generate(start, n):
    cur = stoi[start]
    out = [start]
    for _ in range(n):
        nxt = int(torch.multinomial(probs[cur], num_samples=1).item())
        out.append(vocab[nxt])
        cur = nxt
    return "".join(out)


print("")
print("STEP 5: generate text from the model you just built (seeded, reproducible)")
for _ in range(3):
    print('  "%s"' % generate("t", 24))
print("")
print("It is not Shakespeare, but it learned real structure: 'th', 'the', spaces, periods.")
print("The whole model is one count tensor. Next you replace counting with LEARNING.")
