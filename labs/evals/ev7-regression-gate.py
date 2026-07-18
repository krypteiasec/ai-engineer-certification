#!/usr/bin/env python3
"""
LAB EV7: Regression gates in CI.

The reason you build a golden set and graders is this moment: a change ships, the
eval runs automatically, and if the score dropped, the gate BLOCKS the release.
Without it a quiet quality regression reaches users and you find out from a
complaint. In this lab you score a good system under test against a golden set to
set a baseline, then someone "improves" the prompt and accidentally breaks it. You
score the regressed system with the same grader and the gate fires, refusing to
promote the drop. Catching the regression is the success condition, so the lab
exits 0 when the gate correctly says NO.

Run: python3 modules/academy-content/labs/evals/ev7-regression-gate.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

GOLDEN = [
    ("I love this",            "POSITIVE"),
    ("this is terrible",        "NEGATIVE"),
    ("the worst awful product", "NEGATIVE"),
    ("great and nice work",     "POSITIVE"),
    ("amazing and wonderful",   "POSITIVE"),
]

# GOOD system: a proper few-shot sentiment prompt. Returns correct uppercase labels.
def good_system(review):
    return complete("Classify the sentiment of this review.\n"
                    "love -> POSITIVE\nhate -> NEGATIVE\n"
                    f"Input: {review}\nOutput:")

# REGRESSED system: a well-meaning edit dropped the word "sentiment" and the
# examples, so the model no longer classifies at all. A real, subtle regression.
def regressed_system(review):
    return complete(f"Classify this review.\nInput: {review}\nOutput:")

def exact_match_score(system):
    hits = sum(1 for review, gold in GOLDEN if system(review) == gold)
    return hits / len(GOLDEN)

baseline = exact_match_score(good_system)
candidate = exact_match_score(regressed_system)

print("STEP 1: score the current good system to set the baseline")
print(f"  baseline score : {baseline:.2f}")
print("")
print("STEP 2: score the changed (regressed) system with the same grader")
print(f"  candidate score : {candidate:.2f}")

# THE REGRESSION GATE: promotion is allowed only if the new score is at least the
# baseline. A drop must fail the gate.
def regression_gate(candidate_score, baseline_score):
    return candidate_score >= baseline_score   # True = promote, False = block

promote = regression_gate(candidate, baseline)
print("")
print("STEP 3: the gate decides promote or block")
print(f"  candidate >= baseline : {promote}  -> {'PROMOTE' if promote else 'BLOCK the release'}")

# Success = the gate BLOCKED a real drop. The regressed system must score lower,
# and the gate must refuse to promote it.
ok = (candidate < baseline) and (promote is False)
print("")
print(f"REGRESSION GATE CAUGHT THE DROP: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A gate turns a silent regression into a failed build. Next: tracing and observability.")
