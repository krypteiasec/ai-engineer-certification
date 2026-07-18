#!/usr/bin/env python3
"""
LAB PR2: Quantization. Running models cheaply.

Weights ship in 32-bit floats by default. Most of that precision is wasted: a
model does not need 32 bits to know a weight is roughly 0.42. Quantization stores
each weight in fewer bits, and int8 is the workhorse. You map the float range of a
tensor onto the 256 integer values -128..127 with a single scale factor, store the
int8 codes plus that one float scale, and dequantize back when you compute. Memory
drops about 4x (32 bits to 8 bits) and, done per-tensor with a good scale, the
accuracy loss stays small and bounded. This is exactly the idea behind the int8,
AWQ, and GGUF quantized weights an ML team actually serves; AWQ and GGUF just pick
smarter scales and group sizes to push toward 4-bit while holding quality.

This lab quantizes a real torch weight tensor to int8 BY HAND (symmetric,
per-tensor), measures the byte size before and after, and asserts the round-trip
error stays under a strict bound. No approximations you cannot check.

Run: python3 modules/academy-content/labs/production/pr2-quantization.py
"""
import sys
import torch


def quantize_int8(w):
    """Symmetric per-tensor int8 quantization. Returns (codes int8, scale float).
    scale maps the tensor's max magnitude onto 127, so every weight fits -127..127."""
    max_abs = w.abs().max()
    scale = (max_abs / 127.0).clamp(min=1e-8)     # one float scale for the tensor
    codes = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return codes, scale


def dequantize_int8(codes, scale):
    """Reconstruct approximate float weights from int8 codes and the scale."""
    return codes.to(torch.float32) * scale


def main():
    torch.manual_seed(7)
    # a realistic weight matrix: a linear layer, normal-distributed weights.
    W = torch.randn(512, 512, dtype=torch.float32)

    codes, scale = quantize_int8(W)
    W_hat = dequantize_int8(codes, scale)

    # ---- size accounting (real bytes, from the tensors themselves) ----
    fp32_bytes = W.numel() * W.element_size()                 # 4 bytes per weight
    int8_bytes = codes.numel() * codes.element_size() + scale.numel() * scale.element_size()
    ratio = fp32_bytes / int8_bytes

    print("STEP 1: quantize a %dx%d float32 weight tensor to int8" % tuple(W.shape))
    print("  fp32 size : %d bytes (%.1f KB)" % (fp32_bytes, fp32_bytes / 1024))
    print("  int8 size : %d bytes (%.1f KB), incl. one float scale" % (int8_bytes, int8_bytes / 1024))
    print("  size cut  : %.2fx smaller" % ratio)

    # ---- accuracy accounting (bounded error is the whole promise) ----
    abs_err = (W - W_hat).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    # relative error against the tensor's dynamic range: must stay small.
    rel_err = mean_err / (W.abs().mean().item() + 1e-9)

    print("")
    print("STEP 2: measure the round-trip (dequantized) error")
    print("  max abs error  : %.5f" % max_err)
    print("  mean abs error : %.5f" % mean_err)
    print("  mean rel error : %.2f%%" % (rel_err * 100))

    # A functional check: run the same input through fp32 and int8 weights and
    # confirm the outputs stay close. This is what actually matters in serving.
    x = torch.randn(8, 512)
    y_fp32 = x @ W.t()
    y_int8 = x @ W_hat.t()
    out_rel = ((y_fp32 - y_int8).norm() / (y_fp32.norm() + 1e-9)).item()
    print("  output rel error on a forward pass: %.2f%%" % (out_rel * 100))

    # ---- proofs ----
    size_ok = ratio > 3.5                 # ~4x, allowing for the tiny scale float
    error_bounded = rel_err < 0.02 and out_rel < 0.02   # under 2%: bounded loss
    ok = size_ok and error_bounded
    print("")
    print("  ~4x smaller: %s   bounded error (<2%%): %s"
          % ("YES" if size_ok else "NO", "YES" if error_bounded else "NO"))
    print("")
    print("INT8 QUANT CUT SIZE ~4x WITH BOUNDED ERROR: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Cheaper weights, kept quality. Next: the memory the KV cache costs at serve time.")


if __name__ == "__main__":
    main()
