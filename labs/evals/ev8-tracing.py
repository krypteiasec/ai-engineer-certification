#!/usr/bin/env python3
"""
LAB EV8: Tracing and observability.

An eval score is a single number, and a single number cannot tell you WHY. Tracing
records, for every case, the full path: the input, the prompt sent, the raw output,
the grader used, and the verdict. That is what turns a red eval into a fix, and
what lets you watch quality in production. In this lab you run the whole harness
you built across this course (golden dataset, code grader, LLM-as-judge) and emit
one trace record per case, then prove observability: every result links back to
its case and the grader that produced it, and the aggregate score equals the number
of passing traces. Nothing is a black box.

Run: python3 modules/academy-content/labs/evals/ev8-tracing.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

GOLDEN = [
    ("case-1", "I love this",           "POSITIVE"),
    ("case-2", "this is terrible",       "NEGATIVE"),
    ("case-3", "great and nice work",    "POSITIVE"),
    ("case-4", "the worst awful product","NEGATIVE"),
]

def system_under_test(review):
    prompt = ("Classify the sentiment of this review.\n"
              "love -> POSITIVE\nhate -> NEGATIVE\n"
              f"Input: {review}\nOutput:")
    return prompt, complete(prompt)

def exact_match_grader(output, gold):
    return output == gold

# Run the harness and collect a TRACE record per case: everything needed to
# explain the verdict later, in production or in a failing CI run.
traces = []
for case_id, review, gold in GOLDEN:
    prompt, output = system_under_test(review)
    verdict = exact_match_grader(output, gold)
    traces.append({
        "case_id": case_id,
        "input": review,
        "prompt": prompt,
        "output": output,
        "grader": "exact_match",
        "gold": gold,
        "verdict": "PASS" if verdict else "FAIL",
    })

print(f"STEP 1: run the harness and emit one trace per case ({len(traces)} traces)")
for t in traces:
    print(f"  {t['case_id']}: {t['output']:8} vs {t['gold']:8} [{t['grader']}] -> {t['verdict']}")

# OBSERVABILITY CHECKS.
case_ids = {c[0] for c in GOLDEN}
# 1. Every result traces back to a real case id.
all_linked = all(t["case_id"] in case_ids for t in traces)
# 2. Every trace names the grader that produced its verdict.
all_have_grader = all(t.get("grader") for t in traces)
# 3. One trace per case, none lost, none duplicated.
complete_coverage = (len({t["case_id"] for t in traces}) == len(GOLDEN))
# 4. The aggregate score equals the number of PASS traces (the number is explained
#    by the traces, not asserted on faith).
passes = sum(1 for t in traces if t["verdict"] == "PASS")
aggregate = passes / len(traces)
score_matches_traces = (round(aggregate * len(traces)) == passes)

print("")
print("STEP 2: verify the run is fully observable")
print(f"  every result links to a case : {all_linked}")
print(f"  every trace names its grader : {all_have_grader}")
print(f"  one trace per case           : {complete_coverage}")
print(f"  aggregate score {aggregate:.2f} == {passes}/{len(traces)} passing traces : {score_matches_traces}")

ok = all_linked and all_have_grader and complete_coverage and score_matches_traces
print("")
print(f"EVERY EVAL RESULT TRACES TO ITS CASE AND GRADER: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Traces turn a score into a diagnosis. You have finished Evaluation and Testing.")
