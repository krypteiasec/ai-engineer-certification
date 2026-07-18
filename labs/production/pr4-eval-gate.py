#!/usr/bin/env python3
"""
LAB PR4: Evals as a production gate. Eval-gated CI/CD.

In a mature LLM system, no version reaches users because it "looked good in the
chat window." It reaches users because it PASSED the eval gate. You keep a golden
test set with known-correct answers, you score every candidate against it, and
your deploy pipeline promotes the candidate only if its score clears a threshold.
A candidate that regresses gets BLOCKED before a single real user sees it. This is
the same gate the Evaluation course builds, now wired into a release pipeline: the
eval score is the merge button.

This lab serves a golden set through two candidate configurations. The GOOD
candidate carries the correct classification instruction and scores high. The
REGRESSED candidate has had its instruction dropped (a realistic prompt-edit
regression), so it falls back to echoing and scores low. The pipeline gates both
on the same threshold and proves it passes the good one and blocks the bad one.

Run: python3 modules/academy-content/labs/production/pr4-eval-gate.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# ---- the golden set: reviews with human-validated labels ----
GOLDEN = [
    ("I love this product, it works perfectly", "positive"),
    ("this is terrible and broken", "negative"),
    ("the app is fantastic and I recommend it", "positive"),
    ("worst purchase, totally useless", "negative"),
    ("an awesome and brilliant experience", "positive"),
    ("boring, slow, and disappointing", "negative"),
]

THRESHOLD = 0.80   # a candidate must score at least this to be promoted


def score_candidate(instruction):
    """Serve every golden input through a candidate's instruction and return the
    fraction it labels correctly. The instruction IS the candidate config."""
    correct = 0
    for review, gold in GOLDEN:
        # the candidate's prompt = its instruction + the input under test.
        out = complete(instruction + " Input: " + review)
        if out.strip().lower() == gold:
            correct += 1
    return correct / len(GOLDEN)


def deploy_gate(name, instruction):
    """The pipeline step: score, compare to threshold, decide promote/block."""
    s = score_candidate(instruction)
    promoted = s >= THRESHOLD
    print("  candidate %-10s score %.2f  threshold %.2f  ->  %s"
          % (name, s, THRESHOLD, "PROMOTE" if promoted else "BLOCK"))
    return s, promoted


def main():
    print("STEP 1: score two candidates against the golden set through the gate")
    # GOOD: correct instruction, the model classifies sentiment properly.
    good_score, good_promoted = deploy_gate("good", "Classify the sentiment.")
    # REGRESSED: instruction dropped, so the model has no task and echoes -> wrong.
    bad_score, bad_promoted = deploy_gate("regressed", "Here is a review.")

    print("")
    print("STEP 2: the gate promotes only what clears the threshold")
    print("  good candidate promoted     : %s" % ("YES" if good_promoted else "NO"))
    print("  regressed candidate promoted: %s" % ("YES" if bad_promoted else "NO"))

    # ---- proofs: the good one PASSES, the regressed one is BLOCKED ----
    gate_blocked_regression = (good_promoted is True) and (bad_promoted is False)
    scores_ordered = good_score > bad_score
    ok = gate_blocked_regression and scores_ordered
    print("")
    print("EVAL GATE BLOCKED THE REGRESSED DEPLOY: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("The eval score is the merge button. Next: watch the cost of what you ship.")


if __name__ == "__main__":
    main()
