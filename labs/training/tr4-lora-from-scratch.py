#!/usr/bin/env python3
"""
LAB TR4: LoRA from scratch, and proving the base stays frozen.

Full fine-tuning updates every weight, which is expensive and needs a fresh copy
of the whole model per task. LoRA (Low-Rank Adaptation) is the cheaper idea that
runs the industry: freeze the original weights, and learn a tiny low-rank
correction on top. For a weight matrix W, LoRA adds B @ A where A and B are
skinny (rank r), so you train a few thousand numbers instead of millions. QLoRA
is the same idea with the frozen base kept in 4-bit to save even more memory.

This lab writes the LoRA layer BY HAND so the mechanism is not a mystery:
  output = frozen_base(x) + scaling * (x @ A^T @ B^T)
It wraps a tiny model's linear layers, freezes the base, trains only the adapters
on a task, and PROVES the claim that makes LoRA safe and cheap: only the adapter
parameters carried gradients, and every base weight came out byte-for-byte
unchanged. B starts at zero, so the adapter is a no-op at init and only moves the
model as far as training pushes it.

Run: python3 modules/academy-content/labs/training/tr4-lora-from-scratch.py
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


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear and add a trainable low-rank update B @ A.

    The base weight is frozen (requires_grad = False). Only A and B learn. B is
    initialized to zero so at step 0 the wrapped layer is identical to the base:
    LoRA never damages the starting model, it only adds a learned correction.
    """
    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, base_linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_linear.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A.t()) @ self.B.t())


class TinyNet(nn.Module):
    """A tiny two-linear model. We LoRA-wrap both linears."""
    def __init__(self, d=16):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)

    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))


def inject_lora(model, r=4, alpha=8):
    for p in model.parameters():
        p.requires_grad = False
    model.l1 = LoRALinear(model.l1, r, alpha)
    model.l2 = LoRALinear(model.l2, r, alpha)
    adapters = [model.l1.A, model.l1.B, model.l2.A, model.l2.B]
    return adapters


def main():
    torch.manual_seed(1234)
    D = 16
    model = TinyNet(D)

    # snapshot the base weights BEFORE injecting LoRA, to prove they never move.
    base_before = {
        "l1.w": model.l1.weight.detach().clone(),
        "l2.w": model.l2.weight.detach().clone(),
    }

    adapters = inject_lora(model, r=4, alpha=8)

    # a tiny regression task so there is a real gradient to follow.
    X = torch.randn(64, D)
    target = torch.randn(64, D)

    n_adapter = sum(p.numel() for p in adapters)
    n_total = sum(p.numel() for p in model.parameters())
    print("STEP 1: LoRA-wrap the model and freeze the base")
    print("  trainable adapter params: %d of %d (%.1f%%)"
          % (n_adapter, n_total, 100.0 * n_adapter / n_total))

    # confirm at init the wrapped model equals the base (B = 0 => no-op).
    with torch.no_grad():
        # reconstruct the pure-base output by zeroing scaling contribution: B is 0.
        noop_at_init = bool(torch.allclose(model.l1.B, torch.zeros_like(model.l1.B)))
    print("  adapter is a no-op at init (B starts at zero): %s" % ("YES" if noop_at_init else "NO"))

    opt = torch.optim.AdamW(adapters, lr=1e-2)
    print("")
    print("STEP 2: train ONLY the adapters")
    losses = []
    for step in range(300):
        loss = F.mse_loss(model(X), target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 60 == 0 or step == 299:
            print("  step %3d  loss %.4f" % (step, loss.item()))

    # ---- proofs ----
    # (a) every base parameter has requires_grad False and received no gradient.
    base_frozen = (not model.l1.base.weight.requires_grad
                   and not model.l2.base.weight.requires_grad
                   and model.l1.base.weight.grad is None
                   and model.l2.base.weight.grad is None)
    # (b) base weights are byte-for-byte unchanged after training.
    base_unchanged = (torch.equal(model.l1.base.weight, base_before["l1.w"])
                      and torch.equal(model.l2.base.weight, base_before["l2.w"]))
    # (c) the adapters actually moved (B left zero, A left near-zero would be a dud).
    adapters_moved = (model.l1.B.abs().sum().item() > 0 and model.l2.B.abs().sum().item() > 0)
    # (d) training worked: loss fell.
    loss_fell = losses[-1] < losses[0]

    ok = base_frozen and base_unchanged and adapters_moved and loss_fell
    print("")
    print("  base params frozen (no grad): %s" % ("YES" if base_frozen else "NO"))
    print("  base weights byte-for-byte unchanged: %s" % ("YES" if base_unchanged else "NO"))
    print("  adapters moved and loss fell (%.4f -> %.4f): %s"
          % (losses[0], losses[-1], "YES" if (adapters_moved and loss_fell) else "NO"))
    print("")
    print("ONLY LORA PARAMS TRAINED (base frozen): %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
