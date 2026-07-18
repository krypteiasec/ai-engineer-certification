#!/usr/bin/env python3
"""
LAB RAG8: Evaluating a RAG system.

"It looks like it works" is not evaluation. RAG has two halves and you measure
them separately. RETRIEVAL quality: did the right passage come back? The standard
metric is recall@k, the fraction of questions for which the gold passage is
somewhere in the top-k results. ANSWER quality: was the final answer actually
grounded in what we retrieved? You need a labeled eval set, a small list of
(question, gold passage) pairs, and you score against it every time you change
the system so a regression shows up as a number, not a user complaint. This lab
builds an eval set, computes recall@1 and recall@3, and checks answer grounding.

Run: python3 modules/academy-content/labs/rag/rag8-evaluation.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine, complete

CORPUS = [
    "Honeybees collect nectar from flowers and turn it into honey inside the hive.",
    "The Eiffel Tower in Paris was completed in 1889 for the World Fair.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Mount Everest is the highest mountain above sea level on Earth.",
]
VECTORS = [embed(t) for t in CORPUS]

# The eval set: (question, gold passage id). This is the labeled ground truth you
# score against on every change. Keep it small, real, and honest.
EVAL = [
    ("how do bees make honey from flowers", 0),
    ("when was the eiffel tower built", 1),
    ("how do plants use sunlight energy", 2),
    ("what is the deepest ocean", 3),
    ("the highest mountain on earth", 4),
]


def retrieve(query, k):
    qv = embed(query)
    return sorted(range(len(CORPUS)), key=lambda i: -cosine(qv, VECTORS[i]))[:k]


def recall_at_k(k):
    """Fraction of eval questions whose gold passage is in the top-k."""
    hits = sum(1 for query, gold in EVAL if gold in retrieve(query, k))
    return hits / len(EVAL)


print("STEP 1: score RETRIEVAL against the eval set")
r1 = recall_at_k(1)
r3 = recall_at_k(3)
for query, gold in EVAL:
    got = retrieve(query, 3)
    mark = "hit " if gold in got else "MISS"
    print(f"  [{mark}] gold={gold} top3={got}  {query!r}")
print(f"  recall@1 = {r1:.2f}")

print("")
print("STEP 2: score ANSWER grounding (answer must come from retrieved context)")
grounded_count = 0
for query, gold in EVAL:
    ctx = " ".join(CORPUS[i] for i in retrieve(query, 3))
    ans = complete(f"Context: {ctx}\nQuestion: {query}")
    is_grounded = ans in ctx and ans.strip() != ""
    grounded_count += 1 if is_grounded else 0
answer_grounding = grounded_count / len(EVAL)
print(f"  grounded answers = {grounded_count}/{len(EVAL)} ({answer_grounding:.2f})")

# The headline metric this lab proves: every question retrieves its gold passage
# within the top 3. A regression would drop this below 1.00 and fail the gate.
print("")
print(f"RECALL@3 = {r3:.2f}")
ok = (r3 == 1.00) and (answer_grounding == 1.00)
if not ok:
    print("EVAL GATE: FAIL")
    sys.exit(1)
print("EVAL GATE: PASS (retrieval and grounding both perfect on the eval set)")
print("")
print("You can now measure RAG, not just hope it works. That closes the course:")
print("embeddings, search, chunking, pipeline, generation, failure modes, evaluation.")
