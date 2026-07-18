#!/usr/bin/env python3
"""
LAB TR3: Supervised fine-tuning (SFT), the loop that adapts a model to a task.

Supervised fine-tuning is the workhorse of making a model yours. You collect
examples of the behavior you want (here, a small set of question/answer lines),
then run the exact training loop from Course 1 over them: forward, compute the
loss, backward for gradients, step the optimizer, repeat, and watch the loss
fall. When the loss falls, the model is fitting your examples, and it starts
producing the taught answers it could not produce before.

This lab does real SFT on a tiny GPT (the shared Academy architecture, with
attention, so it can actually learn the mapping):
  1. take a fresh model and confirm it does NOT know the answer yet,
  2. run the SFT loop over the examples across many steps,
  3. watch the loss fall, then confirm the model now completes a held prompt with
     the correct taught answer,
and PROVES both: the SFT loss decreased and the model learned the taught answer.

Tiny and fast: a small model, a few hundred steps, a second or two.

Run: python3 modules/academy-content/labs/training/tr3-sft.py
"""
import sys, os
_labs = None
for _c in [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
           os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]:
    if _c and os.path.isdir(os.path.join(_c, "_models")): _labs = os.path.abspath(_c); break
if _labs: sys.path.insert(0, os.path.join(_labs, "_models"))

import torch
import torch.nn.functional as F
from model import TinyGPT, CharTokenizer, generate

# The supervised examples: the task we want the model to learn. Repeated so a
# tiny model sees each mapping many times, exactly like a real SFT dataset.
EXAMPLES = [
    "q: capital of france? a: paris.",
    "q: capital of japan? a: tokyo.",
    "q: capital of italy? a: rome.",
    "q: capital of spain? a: madrid.",
]
TEXT = ("\n".join(EXAMPLES) + "\n") * 6
TOK = CharTokenizer(sorted(set(TEXT)))
V = TOK.vocab_size
BLOCK = 32
PROMPT = "q: capital of japan? a:"


def batch(data, bs, gen):
    ix = torch.randint(0, len(data) - BLOCK - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


def answer_for(model):
    """Greedily complete the held prompt and return just the answer it produced."""
    out = generate(model, TOK, PROMPT, n=8, temperature=0.0, device="cpu")
    return out[len(PROMPT):].strip()


def main():
    torch.manual_seed(1234)
    gen = torch.Generator().manual_seed(1234)
    data = torch.tensor(TOK.encode(TEXT), dtype=torch.long)
    model = TinyGPT(V, c=64, n_layer=2, n_block=BLOCK)

    before = answer_for(model)  # fresh model, before any fine-tuning

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("STEP 1-4: the SFT loop (forward, loss, backward, step) over epochs")
    losses = []
    STEPS = 400
    for step in range(STEPS):
        x, y = batch(data, 24, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print("  step %3d  loss %.4f" % (step, loss.item()))

    after = answer_for(model)
    start = sum(losses[:10]) / 10
    end = sum(losses[-10:]) / 10
    decreased = end < start * 0.5
    learned_task = ("tokyo" in after) and ("tokyo" not in before)

    print("")
    print("  loss (first 10 avg) %.4f -> (last 10 avg) %.4f" % (start, end))
    print("  before SFT, prompt %r -> answer %r" % (PROMPT, before[:12]))
    print("  after  SFT, prompt %r -> answer %r" % (PROMPT, after[:12]))
    print("  model learned the taught answer (says tokyo now, did not before): %s"
          % ("YES" if learned_task else "NO"))
    print("")
    print("SFT LOSS DECREASED: %s" % ("YES" if decreased else "NO"))
    if not (decreased and learned_task):
        sys.exit(1)


if __name__ == "__main__":
    main()
