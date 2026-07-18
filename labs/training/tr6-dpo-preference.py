#!/usr/bin/env python3
"""
LAB TR6: Preference tuning (DPO), teaching a model what people prefer.

SFT teaches a model to imitate example answers. But often you cannot write the
one right answer, you can only say which of two answers is BETTER. Preference
tuning learns from those comparisons. Direct Preference Optimization (DPO) is the
method that made this simple: no separate reward model, just pairs of
(prompt, chosen, rejected), and a loss that raises the model's probability of the
CHOSEN response over the REJECTED one, measured against a FROZEN reference copy of
the model so it does not drift too far from where it started.

This lab runs real DPO on a tiny GPT:
  1. build a tiny policy model and a frozen reference (a copy of it at start),
  2. for each pair, score the chosen and rejected responses under both models,
  3. apply the DPO loss  -log sigmoid( beta * (policy_margin - reference_margin) ),
  4. train, and PROVE the chosen-over-rejected margin rose and every pair now
     prefers the chosen response.

Tiny and fast: a small model, a few hundred steps, a second or two.

Run: python3 modules/academy-content/labs/training/tr6-dpo-preference.py
"""
import sys, os, copy
_labs = None
for _c in [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
           os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]:
    if _c and os.path.isdir(os.path.join(_c, "_models")): _labs = os.path.abspath(_c); break
if _labs: sys.path.insert(0, os.path.join(_labs, "_models"))

import torch
import torch.nn.functional as F
from model import TinyGPT, CharTokenizer

# Preference pairs: same prompt, a preferred (chosen) and a dispreferred (rejected)
# continuation. The model should learn to favor the chosen style.
PAIRS = [
    ("request: ", "please, thank you.", "no, go away now."),
    ("reply:   ", "sure, happy to help.", "figure it out yourself."),
    ("answer:  ", "yes, here is the plan.", "not my problem at all."),
]
ALL = "".join(p + c + r for p, c, r in PAIRS)
VOCAB = sorted(set(ALL))
TOK = CharTokenizer(VOCAB)
BETA = 0.1


def seq_logprob(model, prompt, response):
    """Teacher-forced log-probability of `response` given `prompt`, summed over the
    response tokens only. This is the score DPO compares between chosen/rejected."""
    p_ids = TOK.encode(prompt)
    full = TOK.encode(prompt + response)
    x = torch.tensor(full, dtype=torch.long)
    logp = F.log_softmax(model(x), dim=-1)  # (T, V)
    total = 0.0
    for i in range(len(p_ids), len(full)):
        total = total + logp[i - 1, full[i]]
    return total


def margin(model):
    """Mean (logp(chosen) - logp(rejected)) across all pairs, higher = model
    prefers the chosen responses more."""
    with torch.no_grad():
        vals = [seq_logprob(model, p, c) - seq_logprob(model, p, r) for p, c, r in PAIRS]
    return float(sum(vals) / len(vals))


def all_prefer_chosen(model):
    with torch.no_grad():
        return all(seq_logprob(model, p, c) > seq_logprob(model, p, r) for p, c, r in PAIRS)


def main():
    torch.manual_seed(1234)
    # a genuinely tiny GPT (from the shared Academy architecture), 1 block, width 32.
    policy = TinyGPT(TOK.vocab_size, c=32, n_layer=1, n_block=48)
    reference = copy.deepcopy(policy)  # frozen snapshot of the starting model
    for p in reference.parameters():
        p.requires_grad = False

    m0 = margin(policy)
    print("STEP 1: before DPO")
    print("  chosen-over-rejected margin: %.4f" % m0)
    print("  every pair already prefers chosen: %s" % ("YES" if all_prefer_chosen(policy) else "NO"))

    opt = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    print("")
    print("STEP 2: DPO training (raise chosen over rejected vs the frozen reference)")
    for step in range(250):
        loss = 0.0
        for p, c, r in PAIRS:
            pi_c = seq_logprob(policy, p, c)
            pi_r = seq_logprob(policy, p, r)
            with torch.no_grad():
                ref_c = seq_logprob(reference, p, c)
                ref_r = seq_logprob(reference, p, r)
            # DPO loss: prefer chosen, but stay anchored to the reference model.
            logits = BETA * ((pi_c - pi_r) - (ref_c - ref_r))
            loss = loss - F.logsigmoid(logits)
        loss = loss / len(PAIRS)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == 249:
            print("  step %3d  dpo loss %.4f  margin %.4f" % (step, loss.item(), margin(policy)))

    m1 = margin(policy)
    rose = m1 > m0
    prefer = all_prefer_chosen(policy)
    ok = rose and prefer
    print("")
    print("  margin before %.4f -> after %.4f (rose: %s)" % (m0, m1, "YES" if rose else "NO"))
    print("  every pair now prefers the chosen response: %s" % ("YES" if prefer else "NO"))
    print("")
    print("PREFERENCE TUNING RAISED CHOSEN-OVER-REJECTED MARGIN: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
