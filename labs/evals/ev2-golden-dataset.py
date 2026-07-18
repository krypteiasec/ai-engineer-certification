#!/usr/bin/env python3
"""
LAB EV2: Building a golden dataset.

An eval is only as good as the dataset behind it. A GOLDEN dataset is a small,
human-validated set of (input, expected) pairs that you deliberately aim at the
places your system fails: negation, mixed signals, edge cases, not just the easy
wins. In this lab you build one for a sentiment task, then VALIDATE it the way you
would validate any test fixture: every case has both fields, no duplicate inputs,
and every expected label is inside the allowed label set. A malformed golden set
silently corrupts every score you compute from it, so you check it first.

Run: python3 modules/academy-content/labs/evals/ev2-golden-dataset.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

# The golden dataset: (input, expected label, failure-mode tag). The tag records
# WHY each case is here, so the set stays aimed at real weaknesses over time.
GOLDEN = [
    ("I love this, it works perfectly",   "POSITIVE", "easy-positive"),
    ("this is a terrible broken mess",     "NEGATIVE", "easy-negative"),
    ("the product is fine, nothing special","NEUTRAL",  "edge-neutral"),
    ("not good at all, very disappointing", "NEGATIVE", "hard-negation"),
    ("great idea but the app is buggy",     "NEGATIVE", "hard-mixed"),
    ("the worst, awful and useless",        "NEGATIVE", "easy-negative"),
]

print(f"STEP 1: golden dataset with {len(GOLDEN)} human-validated cases")
for inp, exp, tag in GOLDEN:
    print(f"  [{tag:14}] {exp:8} <- {inp!r}")

# 1. Every case is well formed: it has a non-empty input and a valid label.
well_formed = all(inp.strip() and exp in LABELS for inp, exp, _ in GOLDEN)

# 2. Inputs are unique. A duplicate input double-counts one case and skews the score.
inputs = [inp for inp, _, _ in GOLDEN]
unique_inputs = (len(inputs) == len(set(inputs)))

# 3. Every expected label is inside the allowed set (no typos like "POSTIVE").
labels_in_set = all(exp in LABELS for _, exp, _ in GOLDEN)

# 4. It actually targets failure modes, not just easy wins.
hard_cases = sum(1 for _, _, tag in GOLDEN if tag.startswith(("hard", "edge")))

print("")
print("STEP 2: validate the fixture before trusting any score from it")
print(f"  every case well formed   : {well_formed}")
print(f"  inputs unique            : {unique_inputs}")
print(f"  labels inside {sorted(LABELS)} : {labels_in_set}")
print(f"  failure-mode cases       : {hard_cases} of {len(GOLDEN)}")

ok = well_formed and unique_inputs and labels_in_set and hard_cases >= 3
print("")
print(f"GOLDEN DATASET IS VALID ({len(GOLDEN)} cases, unique inputs, labels in set): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A validated golden set is the foundation. Next: code-based graders score against it.")
