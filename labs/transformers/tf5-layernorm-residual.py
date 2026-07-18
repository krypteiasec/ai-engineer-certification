#!/usr/bin/env python3
"""
LAB TF5: Layer norm and residuals, why deep transformers train at all.

Stack many layers and a plain network becomes unstable: each layer multiplies the
signal, so activations either explode toward infinity or collapse toward zero, and
the gradients go with them. Two small tricks tame this and sit in every
transformer block:
  - LAYER NORM rescales each token vector to mean 0 and unit variance, so numbers
    entering a sublayer are always a sane size.
  - the RESIDUAL connection adds the input back, x + sublayer(x), so the signal has
    a clean straight-through path and each block only has to learn a small change.

You build a 40-layer stack twice, once plain and once with pre-norm residuals,
using the same linear sublayer (a matrix multiply, the core of any layer). You
PROVE the plain stack's activations explode while the residual version stays
bounded, and that layer norm really does produce mean ~0, variance ~1.

Run: python3 modules/academy-content/labs/transformers/tf5-layernorm-residual.py
"""
import sys
import torch
import torch.nn as nn

torch.manual_seed(17)  # reproducible

C, DEPTH = 16, 40
x0 = torch.randn(C)

# One fixed weight per layer with gain > 1, so each layer scales the signal up.
# This is exactly the regime that makes naive depth diverge.
GAIN = 1.6
layers = [torch.randn(C, C) * (GAIN / (C ** 0.5)) for _ in range(DEPTH)]
ln = nn.LayerNorm(C)

# PATH A: plain deep stack, x = W @ x, layer after layer. No norm, no residual.
x = x0.clone()
for W in layers:
    x = x @ W
plain_ratio = x.norm().item() / x0.norm().item()
print("STEP 1: plain %d-layer stack, output/input norm ratio = %.3e" % (DEPTH, plain_ratio))

# PATH B: pre-norm residual stack, x = x + sublayer(norm(x)). Same weights.
x = x0.clone()
with torch.no_grad():
    for W in layers:
        x = x + ln(x) @ W
resid_ratio = x.norm().item() / x0.norm().item()
print("STEP 2: pre-norm residual stack, output/input norm ratio = %.3f" % resid_ratio)

# STEP 3: layer norm itself gives each vector mean ~0 and variance ~1.
probe = torch.randn(C) * 40 + 12          # a wildly off-scale vector
with torch.no_grad():
    n = ln(probe)
mean_ok = n.mean().abs().item() < 1e-5
var_ok = abs(n.var(unbiased=False).item() - 1.0) < 1e-2
print("STEP 3: layer norm output mean=%.2e var=%.3f (want 0 and 1): %s" % (
    n.mean().item(), n.var(unbiased=False).item(), "YES" if (mean_ok and var_ok) else "NO"))

# STEP 4: the invariants.
#  (a) the plain stack is unstable: its activations blow up past any sane bound.
#  (b) the residual stack stays bounded: activations grow gently, not explosively.
plain_exploded = plain_ratio > 1e4
resid_bounded = resid_ratio < 50.0
print("STEP 4: plain exploded (ratio > 1e4): %s | residual bounded (ratio < 50): %s" % (
    "YES" if plain_exploded else "NO", "YES" if resid_bounded else "NO"))

ok = plain_exploded and resid_bounded and mean_ok and var_ok
print("")
print("LAYERNORM+RESIDUAL KEEP ACTIVATIONS STABLE THROUGH DEPTH: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("Normalize the input to each sublayer, add the signal back around it, and a")
print("40-layer network stays sane. Without these two tricks, depth does not train.")
