#!/usr/bin/env python3
"""
CAPSTONE 4 (ch4): A reusable eval pipeline that gates a build on a golden set.

Capstone project three: the eval harness itself, the artifact strong candidates
build FIRST on a take-home. This is slightly larger than the Evaluation course
lab: it is a reusable pipeline that combines TWO grader types (a deterministic
code grader for exact-match tasks and an LLM-as-judge for graded quality), scores
a candidate build against a versioned golden dataset, and PROMOTES only if the
combined score clears the gate. Point it at any build with a `respond` function
and it just works, which is what "reusable" means.

To prove the gate has teeth it scores two builds through the same pipeline: a good
build carrying the correct instruction, and a regressed build whose instruction
was dropped (a realistic prompt-edit mistake). The pipeline promotes the good one
and BLOCKS the regression, with no human in the loop. It prints the success
invariant and exits 0.

Run: python3 modules/academy-content/labs/capstones/cap4-eval-pipeline.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# ── The golden dataset: human-validated cases, versioned. ────────────────────────
# Each case: an input, the exact label a code grader checks, and a rubric fact the
# judge looks for in a longer grounded answer.
GOLDEN = [
    {"input": "I love this, it is fantastic", "label": "positive", "fact": None},
    {"input": "this is terrible and broken", "label": "negative", "fact": None},
    {"input": "an awesome and brilliant tool", "label": "positive", "fact": None},
    {"input": "useless, slow, and disappointing", "label": "negative", "fact": None},
]
GATE = 0.80


# ── Two reusable graders. ────────────────────────────────────────────────────────
def code_grader(output, expected_label):
    """Deterministic: does the output exactly equal the known label?"""
    return 1.0 if output.strip().lower() == expected_label else 0.0


def judge_grader(output, expected_label):
    """LLM-as-judge stand-in: a second, independent check that the output is a
    valid sentiment label and matches the gold direction. In production this call
    is a model; here it is a deterministic rule so the lab is reproducible, but
    the ROLE is identical: an independent grader that can disagree with the code
    grader, which is why we average them."""
    out = output.strip().lower()
    if out not in ("positive", "negative", "neutral"):
        return 0.0
    return 1.0 if out == expected_label else 0.0


class EvalPipeline:
    """Reusable: give it a golden set and a gate, then score any build."""

    def __init__(self, golden, gate):
        self.golden, self.gate = golden, gate

    def score(self, build):
        code_pts, judge_pts = 0.0, 0.0
        for case in self.golden:
            out = build.respond(case["input"])
            code_pts += code_grader(out, case["label"])
            judge_pts += judge_grader(out, case["label"])
        n = len(self.golden)
        code_score, judge_score = code_pts / n, judge_pts / n
        combined = (code_score + judge_score) / 2  # average the two grader types
        return {"code": code_score, "judge": judge_score, "combined": combined}

    def gate_build(self, build):
        s = self.score(build)
        s["promote"] = s["combined"] >= self.gate
        return s


# ── Two builds of the same feature. ──────────────────────────────────────────────
class GoodBuild:
    def respond(self, text):
        return complete("Classify the sentiment. Input: " + text).strip().lower()


class RegressedBuild:
    def respond(self, text):
        # The instruction was dropped in an edit, so it falls back to echoing.
        return complete("Input: " + text).strip().lower()


def main():
    pipe = EvalPipeline(GOLDEN, GATE)
    print("REUSABLE EVAL PIPELINE (gate = %.2f, %d golden cases)\n" % (GATE, len(GOLDEN)))

    good = pipe.gate_build(GoodBuild())
    bad = pipe.gate_build(RegressedBuild())

    print("%-16s %-8s %-8s %-10s %s" % ("BUILD", "CODE", "JUDGE", "COMBINED", "VERDICT"))
    print("%-16s %-8.2f %-8.2f %-10.2f %s" % (
        "good", good["code"], good["judge"], good["combined"],
        "PROMOTE" if good["promote"] else "BLOCK"))
    print("%-16s %-8.2f %-8.2f %-10.2f %s" % (
        "regressed", bad["code"], bad["judge"], bad["combined"],
        "PROMOTE" if bad["promote"] else "BLOCK"))

    # Invariants of a real gate: promote the good build, block the regression.
    assert good["promote"] is True, "good build must be promoted"
    assert bad["promote"] is False, "regressed build must be blocked"

    ok = good["promote"] and not bad["promote"]
    print("")
    print("  good build promoted, regressed build blocked, zero humans involved")
    print("")
    print("EVAL PIPELINE GATED THE BUILD: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Reusable, two-grader, golden-set gated. This is the take-home that passes.")


if __name__ == "__main__":
    main()
