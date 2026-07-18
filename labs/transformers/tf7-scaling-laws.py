#!/usr/bin/env python3
"""
LAB TF7: Scaling laws, why loss falls in a straight line on a log-log plot.

The empirical discovery that drove the whole modern era: as you scale a model's
size (or its data, or its compute), the loss does not drop randomly. It follows a
POWER LAW. Loss = A * N^(-alpha), which is a straight line when you plot log(loss)
against log(size). That predictability is what let labs justify spending millions
on a bigger run: you can forecast the loss before you train.

You reproduce the shape from first principles here. Build a target signal whose
frequency components have power-law-decaying strength, then approximate it with
models of increasing capacity N (keeping the first N components). The leftover
error is a clean power law in N. You fit the exponent in log-log space with least
squares and PROVE it matches the known truth and that the fit is near-perfect.

Run: python3 modules/academy-content/labs/transformers/tf7-scaling-laws.py
"""
import sys
import numpy as np

rng = np.random.default_rng(29)  # reproducible

# STEP 1: a target with power-law spectrum. Component k has amplitude k^(-P).
# Truncating to the first N components leaves residual energy sum_{k>N} k^(-2P),
# whose square root (the RMSE) is a power law in N. That is our "loss vs scale".
P = 1.2
KMAX = 4000
k = np.arange(1, KMAX + 1)
amp = k ** (-P)                    # true component strengths
total_energy = np.sum(amp ** 2)
print("STEP 1: target signal with power-law spectrum, exponent P=%.2f, %d components" % (P, KMAX))

# STEP 2: measure loss (residual RMSE) at increasing model capacity N.
capacities = [2, 4, 8, 16, 32, 64, 128, 256, 512]
losses = []
for N in capacities:
    residual_energy = np.sum(amp[N:] ** 2)     # energy in components we dropped
    losses.append(np.sqrt(residual_energy))
losses = np.array(losses)
print("STEP 2: loss vs capacity N (bigger N, lower loss):")
for N, L in zip(capacities, losses):
    print("   N=%4d  loss=%.6f" % (N, L))

# STEP 3: fit a line to log(loss) vs log(N). The slope IS the scaling exponent.
logN = np.log(np.array(capacities, dtype=float))
logL = np.log(losses)
slope, intercept = np.polyfit(logN, logL, 1)
# R^2 of the log-log fit: how straight is the line (how clean is the power law).
pred = slope * logN + intercept
ss_res = np.sum((logL - pred) ** 2)
ss_tot = np.sum((logL - logL.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print("STEP 3: log-log least-squares fit: slope=%.3f  R^2=%.5f" % (slope, r2))

# STEP 4: the invariants.
#  (a) loss strictly decreases with scale (diminishing but real returns).
#  (b) the relationship is a power law: the log-log fit is essentially a line.
#  (c) the fitted exponent matches theory. Truncating a k^(-P) spectrum gives an
#      RMSE tail ~ N^(-(P-0.5)), so the expected slope is -(P-0.5).
expected_slope = -(P - 0.5)
monotonic = bool(np.all(np.diff(losses) < 0))
straight = r2 > 0.99
matches = abs(slope - expected_slope) < 0.06
print("STEP 4: loss monotically falls: %s | log-log is a line (R^2>0.99): %s" % (
    "YES" if monotonic else "NO", "YES" if straight else "NO"))
print("        fitted exponent %.3f vs theory %.3f (match): %s" % (
    slope, expected_slope, "YES" if matches else "NO"))

ok = monotonic and straight and matches
print("")
print("LOSS FOLLOWS A POWER LAW IN SCALE (FITTED EXPONENT MATCHES): %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("A straight line on a log-log plot. That is a scaling law, and its slope lets")
print("you predict the loss of a model you have not trained yet.")
