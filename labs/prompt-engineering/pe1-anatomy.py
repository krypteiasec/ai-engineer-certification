#!/usr/bin/env python3
"""
LAB PE1: Anatomy of a prompt. System, instructions, context.

A prompt is not one blob of text. It has parts: a SYSTEM role that sets the job,
an INSTRUCTION that says what to do, and the INPUT the model acts on. Structure
is what makes a model reliable. In this lab you send the SAME content two ways,
as raw text and as a structured system+instruction+input prompt, and prove the
structured one produces a correct, on-task answer while the raw blob does not.

Run: python3 modules/academy-content/labs/prompt-engineering/pe1-anatomy.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

review = "I love this product, it works perfectly"

# 1. RAW: just hand the model the text with no role, no instruction. It has no
#    task to perform, so it cannot give you a label. It falls back to echoing.
raw = complete(review)
print("STEP 1: raw text, no structure")
print(f"  prompt : {review!r}")
print(f"  output : {raw!r}")

# 2. STRUCTURED: a system role sets the job, an instruction says the task, and
#    the input is clearly separated. This is the anatomy of a real prompt.
structured = chat([
    {"role": "system", "content": "You are a classifier. You label sentiment."},
    {"role": "user", "content": "Classify the sentiment. Input: " + review},
])
print("")
print("STEP 2: system + instruction + input")
print(f"  output : {structured!r}")

# 3. The instruction hierarchy in action: the structured prompt returns the
#    correct label 'positive'; the raw blob never does.
structured_ok = (structured == "positive")
raw_is_label = raw in ("positive", "negative", "neutral")
print("")
print(f"STEP 3: structured gives the correct label : {structured_ok}")
print(f"        raw blob gives a label            : {raw_is_label}")

ok = structured_ok and not raw_is_label
print("")
print(f"STRUCTURE STEERS OUTPUT (system+instruction beats raw text): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A prompt has parts. Structure is the first lever. Next: few-shot examples.")
