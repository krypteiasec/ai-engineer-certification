#!/usr/bin/env python3
"""
LAB MM3: How image generation works, denoising step by step.

Modern image generators (diffusion models) do something that sounds strange and
turns out to be the whole trick: they learn to REMOVE noise. Training corrupts a
real image with more and more random noise until it is static, and the model
learns to undo one step of that. To generate, you start from pure noise and run
the model many times, each pass removing a little noise, until a clean picture
emerges. This lab shows the mechanism with a 1-D "image" (a smooth signal): it
adds heavy noise, then runs a deterministic denoiser for several steps and proves
the error against the clean target falls monotonically, step after step. That
downhill error curve IS the reverse-diffusion process, stripped to its essence.

The denoiser here is a simple low-pass smoother, not a learned network, but the
loop is the real one: start noisy, denoise repeatedly, watch the signal resolve.

Run: python3 modules/academy-content/labs/multimodal/mm3-image-generation.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

import numpy as np

N = 64        # length of the 1-D signal (our tiny "image")
STEPS = 12    # reverse-diffusion steps


def smooth(x):
    """One denoising step: a 3-tap moving average. Averaging neighbors cancels
    high-frequency noise while preserving the underlying smooth shape. A real
    diffusion model replaces this with a learned network, but the ROLE is the
    same: take a noisy signal, return a slightly cleaner one."""
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    padded = np.pad(x, 1, mode="edge")
    return np.convolve(padded, k, mode="valid")


def mse(a, b):
    return float(np.mean((a - b) ** 2))


print("STEP 1: the clean target and its noisy, corrupted version")
t = np.linspace(0.0, 4.0 * np.pi, N).astype(np.float32)
clean = np.sin(t)                                   # the "image" we want back
rng = np.random.default_rng(0)
noise = rng.standard_normal(N).astype(np.float32)   # heavy gaussian noise
x = clean + 1.2 * noise                             # start: pure-ish noise
print(f"  signal length   : {N} samples")
print(f"  start error     : MSE={mse(x, clean):.4f}  (very noisy)")

print("")
print("STEP 2: run the reverse process, one denoising step at a time")
errors = [mse(x, clean)]
for step in range(1, STEPS + 1):
    x = smooth(x)                     # remove a little noise
    e = mse(x, clean)
    errors.append(e)
    bar = "#" * max(1, int(e / errors[0] * 40))
    print(f"  step {step:2d}  MSE={e:.4f}  {bar}")

print("")
print("STEP 3: check the error fell at every step and ended far lower")
# Invariants of a working denoiser / reverse-diffusion loop:
#  (a) the error decreases (or holds) at EVERY step, never rises,
#  (b) the final error is a small fraction of where we started.
monotonic = all(errors[i + 1] <= errors[i] + 1e-6 for i in range(len(errors) - 1))
big_drop = errors[-1] < errors[0] * 0.5
print(f"  monotonic decrease : {monotonic}")
print(f"  final/start ratio  : {errors[-1] / errors[0]:.3f}  (need < 0.50)")

ok = bool(monotonic and big_drop)
print("")
print(f"DENOISER REDUCED NOISE OVER STEPS: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Generation is denoising run in reverse: start from noise, remove a little")
print("at a time, and a coherent signal appears. Real models learn the denoiser")
print("and steer it with a text prompt. Next: the same numbers game for audio.")
