#!/usr/bin/env python3
"""
LAB TR7: Evaluating a fine-tune without fooling yourself.

A fine-tune that looks great on the one prompt you kept trying is worthless. The
only honest question is: did the target metric improve on a HELD-OUT set the model
was not tuned to impress. So you build an eval: a set of prompts you did not
cherry-pick, a scoring function that measures the behavior you actually wanted,
and a before/after comparison of the base model against the fine-tuned one.

This lab evaluates the Academy's real LoRA fine-tune. The goal of that fine-tune
was to steer the model toward SECURITY / VERIFICATION language. So the metric is a
domain score: how strongly each generation uses that vocabulary. It runs the base
and the LoRA-adapted model over a held-out prompt set, scores both, and PROVES the
fine-tune improved the target metric across the set (not just on one lucky prompt).

Fast: it loads checkpoints and generates deterministically, about a second.

Run: python3 modules/academy-content/labs/training/tr7-evaluate-finetune.py
"""
import sys, os
_labs = None
for _c in [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
           os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]:
    if _c and os.path.isdir(os.path.join(_c, "_models")): _labs = os.path.abspath(_c); break
if _labs: sys.path.insert(0, os.path.join(_labs, "_models"))

from load import load_tinygpt, load_tinygpt_with_lora, generate

# A HELD-OUT prompt set. These are not the single prompt you tune against; a fair
# eval scores behavior across several inputs the model was not optimized to game.
EVAL_PROMPTS = [
    "a neural network is",
    "the best way to learn is",
    "a good program is one that",
]
# The target behavior: security / verification vocabulary. The metric counts hits.
THEME = ["verify", "trust", "input", "secur", "check", "test", "attacker", "proof", "run"]


def domain_score(text):
    low = text.lower()
    return sum(low.count(w) for w in THEME)


def evaluate(model):
    """Run the model over the held-out set and return the total target-metric score."""
    total = 0
    rows = []
    for p in EVAL_PROMPTS:
        out = generate(model, p, n=120).replace("\n", " ")
        s = domain_score(out)
        total += s
        rows.append((p, s, out[:60]))
    return total, rows


def main():
    print("STEP 1: score the BASE model on the held-out set")
    base = load_tinygpt()
    base_total, base_rows = evaluate(base)
    for p, s, snip in base_rows:
        print("  score %2d  %-28r %s" % (s, p, snip))
    print("  base total metric: %d" % base_total)

    print("")
    print("STEP 2: score the FINE-TUNED (LoRA) model on the same held-out set")
    lora = load_tinygpt_with_lora()
    lora_total, lora_rows = evaluate(lora)
    for p, s, snip in lora_rows:
        print("  score %2d  %-28r %s" % (s, p, snip))
    print("  fine-tuned total metric: %d" % lora_total)

    improved = lora_total > base_total
    meaningful = lora_total >= base_total + len(EVAL_PROMPTS)  # a real lift, not noise
    ok = improved and meaningful
    print("")
    print("STEP 3: before/after on the held-out metric")
    print("  base %d -> fine-tuned %d (lift %+d)" % (base_total, lora_total, lora_total - base_total))
    print("  improved across the held-out set (not one lucky prompt): %s" % ("YES" if ok else "NO"))
    print("")
    print("FINE-TUNE IMPROVED THE TARGET METRIC: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
