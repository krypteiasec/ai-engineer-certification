#!/usr/bin/env python3
"""
LAB RAG1: Why a model needs external memory.

A trained model only knows what was in its training data, and that knowledge is
frozen. Ask it about YOUR document, a fact from last week, or anything it never
saw, and it cannot answer, it can only guess. Retrieval augmented generation
(RAG) fixes this: keep the facts OUTSIDE the model, fetch the relevant one at
question time, and hand it to the model as context. This lab proves the core
claim in the smallest possible way: the SAME model gives a wrong, ungrounded
answer with no context, and the RIGHT answer once we supply the fact.

Run: python3 modules/academy-content/labs/rag/rag1-external-memory.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine, complete

# Our "external memory": a tiny knowledge base the model was never trained on.
KB = [
    "The Eiffel Tower in Paris was completed in 1889 for the World Fair.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
]

QUESTION = "When was the Eiffel Tower completed?"
FACT = "1889"

print("STEP 1: ask the model WITHOUT giving it the fact")
ungrounded = complete(QUESTION)
print(f"  question : {QUESTION}")
print(f"  answer   : {ungrounded!r}")
print(f"  contains the real answer ({FACT})? {'yes' if FACT in ungrounded else 'no'}")

print("")
print("STEP 2: fetch the relevant fact from external memory, then ask again")
# Retrieval, the honest way: embed the question and every KB entry, keep the
# closest one. (You build the full retriever in later chapters; here we just
# show WHY it matters.)
qv = embed(QUESTION)
best = max(KB, key=lambda passage: cosine(qv, embed(passage)))
print(f"  retrieved: {best!r}")
grounded = complete(f"Context: {best}\nQuestion: {QUESTION}")
print(f"  answer   : {grounded!r}")
print(f"  contains the real answer ({FACT})? {'yes' if FACT in grounded else 'no'}")

# The invariant that defines RAG: external memory turns a guess into a grounded
# answer. The fact must be ABSENT from the frozen answer and PRESENT once we
# retrieve and supply it.
ok = (FACT not in ungrounded) and (FACT in grounded)
print("")
print(f"EXTERNAL MEMORY SUPPLIES THE ANSWER: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("That is the whole idea of RAG. Next: embeddings, the vectors that let us")
print("find the RIGHT passage to retrieve.")
