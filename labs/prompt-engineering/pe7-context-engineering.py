#!/usr/bin/env python3
"""
LAB PE7: Context engineering.

A model can only use what you put in front of it, and every token you send costs
money and adds noise. Context engineering is the discipline of putting the RIGHT
information in front of the model and nothing else: rank your passages by real
similarity to the question, keep only the top ones, and feed that clean, short
context. In this lab you embed and rank candidate passages, feed only the most
relevant one, and prove the trimmed context answers correctly using a fraction of
the tokens a full dump would burn.

Run: python3 modules/academy-content/labs/prompt-engineering/pe7-context-engineering.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

question = "What powers a sailboat?"
passages = [
    "Wind powers a sailboat across the water.",        # relevant
    "A train runs on steel rails and trucks use roads.",  # off-topic
    "The library opens at nine each morning.",         # irrelevant noise
]

# 1. RANK the passages by embedding similarity to the question. This is
#    retrieval: score meaning, then keep only what actually relates to the ask.
qv = embed(question)
ranked = sorted(passages, key=lambda p: cosine(qv, embed(p)), reverse=True)
print("STEP 1: rank passages by similarity to the question")
for p in ranked:
    print(f"  {cosine(qv, embed(p)):.3f}  {p!r}")
top = ranked[0]
retrieval_correct = (top == "Wind powers a sailboat across the water.")

# 2. TRIMMED context: feed the model ONLY the top passage.
trimmed_prompt = f"Context: {top}\nQuestion: {question}"
trimmed = complete(trimmed_prompt)
print("")
print("STEP 2: answer from the TRIMMED context (top passage only)")
print(f"  answer : {trimmed!r}")
trimmed_correct = "wind" in trimmed.lower()

# 3. COST: the trimmed context is a fraction of the size of dumping everything.
trim_tokens = len(("Context: " + top).split())
dump_tokens = len(("Context: " + " ".join(passages)).split())
print("")
print(f"STEP 3: context tokens sent -> trimmed {trim_tokens} vs full dump {dump_tokens}")
cheaper = trim_tokens < dump_tokens

print("")
print(f"        retrieval put the right chunk first : {retrieval_correct}")
print(f"        trimmed context answered correctly  : {trimmed_correct}")
print(f"        trimmed context was cheaper         : {cheaper}")

ok = retrieval_correct and trimmed_correct and cheaper
print("")
print(f"CONTEXT ENGINEERING RETRIEVES THE RIGHT CHUNK: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Feed the model the right context, not all of it. Next: eval-driven tuning.")
