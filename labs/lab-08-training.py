#!/usr/bin/env python3
"""
LAB 08: The training loop: how a model actually learns.

Chapter 3 built a bigram by COUNTING. Now you build the same bigram as a neural
network and LEARN its weights with gradient descent, the exact loop that trains
every LLM, just tiny. The model is one nn.Embedding(V, V): row W[x] holds the
logits over the next character given current character x. The loop:
  1. forward: logits = W[x]  (then cross-entropy applies softmax internally),
  2. loss: F.cross_entropy, how surprised the model is by the true next char,
  3. backward: loss.backward() -- AUTOGRAD computes every gradient for you,
  4. update: optimizer.step() nudges W downhill; optimizer.zero_grad() resets,
  5. repeat, and WATCH THE LOSS FALL. That falling number is learning.

This is the whole point of doing it in PyTorch: you no longer hand-derive the
gradient (as the pure version did). autograd + an optimizer is the real job.

Run: python3 modules/academy-content/labs/lab-08-training.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = "the cat ran. the dog ran. the cat sat. a dog sat. the rat sat."
vocab = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(vocab)}
V = len(vocab)

# training data: every adjacent pair (current -> next) becomes one example.
xs = torch.tensor([stoi[CORPUS[i]] for i in range(len(CORPUS) - 1)], dtype=torch.long)
ys = torch.tensor([stoi[CORPUS[i + 1]] for i in range(len(CORPUS) - 1)], dtype=torch.long)


def train(steps=200, lr=0.5):
    """Train the neural-net bigram with autograd. Returns model + loss history."""
    torch.manual_seed(1234)
    # W is a V x V table: W(x) gives the logits over the next char. nn.Embedding
    # is the clean way to say "look up row x", and its weight is learnable.
    W = nn.Embedding(V, V)
    opt = torch.optim.AdamW(W.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        logits = W(xs)                      # (N, V) forward over the whole batch
        loss = F.cross_entropy(logits, ys)  # softmax + negative-log-likelihood
        opt.zero_grad()                     # clear last step's gradients
        loss.backward()                     # AUTOGRAD fills every .grad
        opt.step()                          # gradient-descent update
        losses.append(loss.item())
    return W, vocab, stoi, losses


if __name__ == "__main__":
    _, _, _, losses = train()
    print("STEP 1-5: training the neural-net bigram with autograd + an optimizer")
    print("  vocab size: %d tokens, %d training pairs" % (V, xs.shape[0]))
    for m in [0, 10, 50, 100, 199]:
        print("  step %3d  loss %.4f" % (m, losses[m]))
    print("")
    dropped = losses[0] - losses[-1]
    learned = losses[-1] < losses[0]
    print("STEP 6: loss fell from %.4f to %.4f  (down %.4f)" % (losses[0], losses[-1], dropped))
    print("        the model LEARNED (final loss < initial loss): %s" % ("YES" if learned else "NO"))
    if not learned:
        sys.exit(1)
    print("")
    print("That falling number is the whole game. A real LLM does this with billions of")
    print("weights, but the loop -- forward, loss, backward, step -- is exactly this one.")
