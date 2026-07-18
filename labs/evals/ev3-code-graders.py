#!/usr/bin/env python3
"""
LAB EV3: Code-based graders.

The cheapest, most reliable grader is plain deterministic code. When there is one
right answer, or a hard format rule, you do not need a model to judge it: you need
an EXACT-MATCH grader and a CONSTRAINT grader. In this lab you run a real system
under test (a few-shot sentiment classifier) across a golden set, score it with an
exact-match grader, and separately check a format constraint (label must be an
uppercase member of the allowed set). The exact-match grader catches the one case
the classifier gets wrong, scoring 4 of 5, exactly as a code grader should.

Run: python3 modules/academy-content/labs/evals/ev3-code-graders.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

# Golden set: five (review, gold label) cases. The negation case is deliberately
# hard, the kind of case a real classifier trips on.
GOLDEN = [
    ("I love this",                 "POSITIVE"),
    ("this is terrible and broken",  "NEGATIVE"),
    ("the worst awful product",      "NEGATIVE"),
    ("great and nice work",          "POSITIVE"),
    ("not good at all",              "NEGATIVE"),   # hard: negation flips meaning
]

# The system under test: a few-shot sentiment classifier built on the mock LLM.
def system_under_test(review):
    prompt = ("Classify the sentiment of this review.\n"
              "love -> POSITIVE\n"
              "hate -> NEGATIVE\n"
              f"Input: {review}\nOutput:")
    return complete(prompt)

# GRADER 1, exact match: is the output exactly the gold label? Deterministic, no model.
def exact_match_grader(output, gold):
    return output == gold

# GRADER 2, constraint: is the output a valid uppercase label at all? Format check.
def constraint_grader(output):
    return output in LABELS

correct = 0
format_ok = 0
print("STEP 1: run the classifier and grade each case with code")
for review, gold in GOLDEN:
    out = system_under_test(review)
    hit = exact_match_grader(out, gold)
    fmt = constraint_grader(out)
    correct += 1 if hit else 0
    format_ok += 1 if fmt else 0
    print(f"  {out:8} gold={gold:8} match={str(hit):5} format_ok={fmt}  <- {review!r}")

total = len(GOLDEN)
print("")
print("STEP 2: aggregate the two code graders")
print(f"  exact-match score : {correct}/{total}")
print(f"  format valid      : {format_ok}/{total}")

# The exact-match grader must score 4/5: four correct, one negation case missed.
ok = (correct == 4) and (total == 5) and (format_ok == total)
print("")
print(f"EXACT-MATCH GRADER SCORED {correct}/{total} CORRECTLY: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Deterministic graders are cheap and exact. Next: judging subjective output with a model.")
