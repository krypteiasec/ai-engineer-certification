#!/usr/bin/env python3
"""
LAB EV4: LLM-as-judge and its calibration.

Some qualities have no exact answer to match: tone, helpfulness, whether a reply
is on brand. For those you use a MODEL as the grader, an LLM-as-judge. But a judge
is itself a fallible model, so the senior skill is not "use a judge", it is
"measure how often the judge agrees with a human, and know where it is unreliable".
In this lab the judge grades support replies PASS or FAIL against a rubric (positive
and helpful passes) using the mock LLM at temperature 0 (deterministic). You then
score the judge against human gold labels and surface the exact case where it is
wrong (negation), which is the calibration signal that tells you not to trust it there.

Run: python3 modules/academy-content/labs/evals/ev4-llm-as-judge.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

# Support replies with HUMAN gold judgments (PASS = positive and helpful).
CASES = [
    ("I would love to help, this is a great question", "PASS"),
    ("that is a terrible broken idea",                  "FAIL"),
    ("happy to sort this out for you",                  "PASS"),
    ("this is awful and useless",                       "FAIL"),
    ("not bad, actually fine",                          "PASS"),   # negation: judge trap
]

# THE LLM-AS-JUDGE. It reads the rubric (encoded as the sentiment demonstrations)
# and returns a verdict, deterministically at temperature 0. This is a model
# grading a model, the core of LLM-as-judge.
def judge(reply):
    prompt = ("Rubric: judge the sentiment of this support reply.\n"
              "great -> POSITIVE\n"
              "awful -> NEGATIVE\n"
              f"Input: {reply}\nOutput:")
    verdict = complete(prompt, temperature=0.0)   # deterministic
    return "PASS" if verdict == "POSITIVE" else "FAIL"

agree = 0
unreliable = []
print("STEP 1: run the judge against human gold labels")
for reply, human in CASES:
    j = judge(reply)
    match = (j == human)
    agree += 1 if match else 0
    if not match:
        unreliable.append(reply)
    print(f"  judge={j:4} human={human:4} agree={str(match):5}  <- {reply!r}")

total = len(CASES)
concordance = agree / total
print("")
print("STEP 2: measure judge-human concordance and flag where the judge is unreliable")
print(f"  concordance         : {agree}/{total} = {concordance:.2f}")
print(f"  unreliable cases    : {unreliable}")

# A useful judge must be measured (concordance computed) AND its failure surfaced.
# Here it agrees on 4 of 5 and is caught being wrong on the negation case.
ok = (agree == 4) and (total == 5) and (len(unreliable) == 1)
print("")
print(f"LLM-AS-JUDGE CONCORDANCE MEASURED ({agree}/{total}), UNRELIABLE CASE FLAGGED: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Never trust a judge you have not calibrated. Next: scoring groundedness and hallucination.")
