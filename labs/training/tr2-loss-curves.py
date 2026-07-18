#!/usr/bin/env python3
"""
LAB TR2: Reading loss curves, overfitting, and validation.

Training loss going down is necessary but not sufficient. The number that tells
you the truth is the VALIDATION loss: the loss on data the model never trained
on. When training loss keeps falling but validation loss starts RISING, the model
has stopped learning the task and started memorizing the training set. That gap
is overfitting, and reading it is the single most important skill in fine-tuning.

This lab trains the same tiny model two ways and reads the curves:
  1. HEALTHY: enough data, modest steps. Train and validation loss both fall.
  2. OVERFIT: a tiny handful of examples trained far too long. Train loss dives
     toward zero while validation loss turns around and climbs.
It then proves a plain overfitting detector (val loss rose from its best) fires
on the overfit run and stays quiet on the healthy one.

Everything is tiny and finishes in a couple of seconds.

Run: python3 modules/academy-content/labs/training/tr2-loss-curves.py
"""
import sys, os
_labs = None
for _c in [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
           os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]:
    if _c and os.path.isdir(os.path.join(_c, "_models")): _labs = os.path.abspath(_c); break
if _labs: sys.path.insert(0, os.path.join(_labs, "_models"))

import torch
import torch.nn as nn
import torch.nn.functional as F

# A small closed vocabulary and a structured corpus a tiny model can learn.
BASE = ("the cat sat on the mat. the dog ran in the sun. a bird sang in the tree. "
        "the fish swam in the sea. the sun set in the west. ")
VOCAB = sorted(set(BASE + "abcdefghijklmnopqrstuvwxyz .,"))
STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
BLOCK = 12


def encode(s):
    return torch.tensor([STOI[c] for c in s if c in STOI], dtype=torch.long)


class TinyLM(nn.Module):
    """A minimal next-token model: embed, one hidden layer, project to logits."""
    def __init__(self, v, c=32):
        super().__init__()
        self.emb = nn.Embedding(v, c)
        self.net = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, v))

    def forward(self, x):
        return self.net(self.emb(x))


def pairs(ids, block):
    x = torch.stack([ids[i:i + block] for i in range(len(ids) - block)])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in range(len(ids) - block)])
    return x, y


def run(train_text, val_text, steps, seed=0):
    torch.manual_seed(seed)
    model = TinyLM(V)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    xt, yt = pairs(encode(train_text), BLOCK)
    xv, yv = pairs(encode(val_text), BLOCK)
    tr, va = [], []
    for step in range(steps):
        logits = model(xt)
        loss = F.cross_entropy(logits.reshape(-1, V), yt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            with torch.no_grad():
                vl = F.cross_entropy(model(xv).reshape(-1, V), yv.reshape(-1)).item()
            tr.append(loss.item()); va.append(vl)
    return tr, va


def overfitting(val_curve, tol=0.02):
    """A plain detector: overfitting shows up as validation loss rising well above
    its best-seen value by the end of training."""
    best = min(val_curve)
    return (val_curve[-1] - best) > tol


def main():
    # HEALTHY: train and validate on different slices of a real corpus.
    long = BASE * 12
    n = int(len(long) * 0.85)
    tr_h, va_h = run(long[:n], long[n:], steps=200, seed=1)
    print("STEP 1: HEALTHY run (enough data, modest steps)")
    print("  train loss: %.3f -> %.3f" % (tr_h[0], tr_h[-1]))
    print("  val   loss: %.3f -> %.3f" % (va_h[0], va_h[-1]))

    # OVERFIT: a tiny training slice trained far too long, judged on unseen text.
    tiny = BASE[:60]
    tr_o, va_o = run(tiny, long[n:], steps=1500, seed=2)
    print("")
    print("STEP 2: OVERFIT run (tiny data, far too many steps)")
    print("  train loss: %.3f -> %.3f  (dives toward memorization)" % (tr_o[0], tr_o[-1]))
    print("  val   loss: %.3f -> %.3f  (turns around and climbs)" % (va_o[0], min(va_o), ))
    print("  val   loss best %.3f -> final %.3f" % (min(va_o), va_o[-1]))

    healthy_ok = not overfitting(va_h)      # healthy run should NOT flag
    overfit_ok = overfitting(va_o)          # overfit run SHOULD flag
    train_fell = tr_h[-1] < tr_h[0]
    print("")
    print("  healthy run flagged as overfitting: %s (want NO)" % ("YES" if overfitting(va_h) else "NO"))
    print("  overfit run flagged as overfitting: %s (want YES)" % ("YES" if overfitting(va_o) else "NO"))
    ok = healthy_ok and overfit_ok and train_fell
    print("")
    print("OVERFITTING SHOWS AS RISING VAL LOSS (detector correct): %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
