#!/usr/bin/env python3
"""
LAB PE5: XML tagging and output shaping.

When a prompt mixes reference material and the actual question in one run-on
blob, the model cannot tell them apart. Delimiters fix this. Tag each part with a
clear label (Context: ... Question: ..., the same idea as <context>...</context>
in a real API) and the model knows what to read and what to answer. In this lab
you ask the same question untagged (ungrounded) and tagged (grounded), and prove
the tags make the model answer from the supplied context.

Run: python3 modules/academy-content/labs/prompt-engineering/pe5-xml-tagging.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

reference = "The moon orbits the earth. The sun is a star. Mars is a red planet."
question = "What orbits the earth?"

# 1. UNTAGGED: reference and question fused into one blob. The model has no
#    markers telling it which part is the source and which is the ask, so it
#    cannot ground its answer.
untagged = complete(reference + " " + question)
print("STEP 1: untagged blob (reference + question fused)")
print(f"  output : {untagged!r}")
untagged_grounded = "moon orbits" in untagged.lower()

# 2. TAGGED: label each part. Context: is the source, Question: is the ask. The
#    model reads the tags and returns the grounded sentence.
tagged = complete(f"Context: {reference}\nQuestion: {question}")
print("")
print("STEP 2: tagged (Context: ... Question: ...)")
print(f"  output : {tagged!r}")
tagged_grounded = "moon orbits" in tagged.lower()

# 3. Tagging is what let the model find and return the right sentence.
print("")
print(f"STEP 3: tagged answer grounded in context   : {tagged_grounded}")
print(f"        untagged answer grounded in context : {untagged_grounded}")

ok = tagged_grounded and not untagged_grounded
print("")
print(f"TAGGING GROUNDS THE OUTPUT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Tags separate source from ask. Next: prompt caching and cost.")
