#!/usr/bin/env python3
"""
LAB EV1: Why evals replace unit tests.

A unit test asserts one exact string. That works for deterministic code and
fails for a language model, because a model can be RIGHT in a hundred different
wordings. "Paris is the capital of France" and "The capital of France is Paris"
are the same answer, but == says they differ. In this lab you build a SEMANTIC
grader: embed both texts and compare direction with cosine against a threshold.
You prove it accepts a valid rewording a unit test rejects, and still rejects a
genuinely wrong answer. That is the whole reason evals exist.

Run: python3 modules/academy-content/labs/evals/ev1-why-evals.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

# The gold answer we accept, and two model outputs to grade against it.
gold = "paris is the capital of france"
valid_variation = "the capital of france is paris"   # correct, reworded
wrong_answer = "berlin is a german city"             # genuinely wrong

# 1. THE UNIT TEST. Exact string equality. Brittle for model output.
def unit_test(answer, gold):
    return answer == gold

# 2. THE SEMANTIC GRADER. Embed both, compare direction with cosine, threshold it.
#    Meaning, not characters. This is the eval mindset in one function.
THRESHOLD = 0.6
def semantic_grader(answer, gold):
    return cosine(embed(answer), embed(gold)) >= THRESHOLD

ut_valid = unit_test(valid_variation, gold)
sg_valid = semantic_grader(valid_variation, gold)
sg_wrong = semantic_grader(wrong_answer, gold)
sim_valid = cosine(embed(valid_variation), embed(gold))
sim_wrong = cosine(embed(wrong_answer), embed(gold))

print("STEP 1: grade a CORRECT but reworded answer")
print(f"  answer          : {valid_variation!r}")
print(f"  unit test (==)  : {ut_valid}   <- rejects a correct answer")
print(f"  semantic (cos>={THRESHOLD}): {sg_valid}   (similarity {sim_valid:.2f})")
print("")
print("STEP 2: grade a WRONG answer with the same semantic grader")
print(f"  answer          : {wrong_answer!r}")
print(f"  semantic (cos>={THRESHOLD}): {sg_wrong}   (similarity {sim_wrong:.2f})")

# 3. The eval is right on both cases the unit test cannot handle: it ACCEPTS the
#    valid variation and REJECTS the wrong answer, where == fails the good one.
ok = (ut_valid is False) and (sg_valid is True) and (sg_wrong is False)
print("")
print(f"SEMANTIC EVAL BEATS THE UNIT TEST (accepts valid variation, rejects wrong): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Model output needs evals, not asserts. Next: build the golden dataset.")
