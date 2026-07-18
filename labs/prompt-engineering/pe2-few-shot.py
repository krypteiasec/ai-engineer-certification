#!/usr/bin/env python3
"""
LAB PE2: Few-shot prompting. Teaching by example.

Zero-shot asks the model to do a task with only an instruction. Few-shot shows
it a handful of worked examples first, and those examples measurably steer both
WHAT it answers and the FORMAT it answers in. In this lab you classify the same
review zero-shot and few-shot, and prove the examples changed the output: the
demonstrated UPPERCASE label format carries straight into the answer.

Run: python3 modules/academy-content/labs/prompt-engineering/pe2-few-shot.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

review = "I love this"

# 1. ZERO-SHOT: instruction only. The model classifies, but in its own default
#    lowercase format.
zero = complete("Classify the sentiment of this review: " + review)
print("STEP 1: zero-shot (instruction only)")
print(f"  output : {zero!r}")

# 2. FEW-SHOT: prepend worked examples. The demonstrated outputs are UPPERCASE,
#    so the model copies that format. Same task, different output.
few = complete(
    "great -> POS\n"
    "awful -> NEG\n"
    "Classify the sentiment of this review: " + review
)
print("")
print("STEP 2: few-shot (examples then the task)")
print("  examples: great -> POS , awful -> NEG")
print(f"  output : {few!r}")

# 3. Prove the examples DID something: the answer is still correct (positive),
#    but the format flipped to match the demonstrations.
same_meaning = (zero == "positive" and few == "POSITIVE")
changed = (zero != few)
print("")
print(f"STEP 3: meaning preserved (both positive) : {same_meaning}")
print(f"        format changed by the examples    : {changed}")

ok = same_meaning and changed
print("")
print(f"FEW-SHOT STEERS OUTPUT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Examples teach format and behavior. Next: chain-of-thought reasoning.")
