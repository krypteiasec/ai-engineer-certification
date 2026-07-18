#!/usr/bin/env python3
"""
LAB PE8: Eval-driven prompt optimization.

You do not improve a prompt by vibes. You define a metric and a test set, score
each candidate prompt against it, and keep the winner. In this lab the downstream
system requires the sentiment label in UPPERCASE and correct. You build a labeled
test set, score a zero-shot prompt and a few-shot prompt against that metric, and
prove the eval objectively selects the higher-scoring prompt instead of a guess.

Run: python3 modules/academy-content/labs/prompt-engineering/pe8-eval-optimization.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

# 1. A labeled TEST SET. The gold label is UPPERCASE, the required output format.
test_set = [
    ("I love this",              "POSITIVE"),
    ("this is great and nice",   "POSITIVE"),
    ("a terrible broken mess",   "NEGATIVE"),
    ("the worst, awful product", "NEGATIVE"),
]
print(f"STEP 1: test set of {len(test_set)} labeled reviews (gold = UPPERCASE label)")

# Two candidate prompt strategies for the same task.
def prompt_zero_shot(review):
    return f"Classify the sentiment of this review.\nInput: {review}\nOutput:"

def prompt_few_shot(review):
    return ("love -> POSITIVE\n"
            "hate -> NEGATIVE\n"
            f"Classify the sentiment of this review.\nInput: {review}\nOutput:")

# 2. The METRIC: fraction of the test set answered correctly AND in the required
#    uppercase format. An exact match against the gold label captures both.
def score(prompt_fn):
    hits = sum(1 for review, gold in test_set if complete(prompt_fn(review)) == gold)
    return hits / len(test_set)

s_zero = score(prompt_zero_shot)
s_few = score(prompt_few_shot)
print("")
print("STEP 2: score each prompt against the metric")
print(f"  zero-shot prompt : {s_zero:.2f}")
print(f"  few-shot prompt  : {s_few:.2f}")

# 3. OPTIMIZE: keep the higher-scoring prompt. The data decides, not opinion.
winner = "few-shot" if s_few > s_zero else "zero-shot"
print("")
print(f"STEP 3: eval selects the winner -> {winner} prompt")

ok = (s_few > s_zero) and (winner == "few-shot")
print("")
print(f"EVAL SELECTED THE HIGHER-SCORING PROMPT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Metric first, then optimize against it. Re-run the eval on every model swap.")
