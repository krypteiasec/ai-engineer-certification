#!/usr/bin/env python3
"""
LAB RAG7: Failure modes, the retrieval miss.

RAG fails in a specific, diagnosable way, and it is almost never the model's
fault first. The classic failure is a RETRIEVAL MISS: the answer is not in the
corpus (or the closest passage is only weakly related), so the top hit comes back
with LOW similarity. Ground a model on an irrelevant passage and you get a
confident, wrong, ungrounded answer, the thing people call "hallucination" but is
really a retrieval problem. The fix is to DETECT the miss with a similarity
threshold and to repair the corpus, not to blame the model. This lab reproduces
the miss, detects it, adds the missing document, and shows the answer recover.

Run: python3 modules/academy-content/labs/rag/rag7-failure-modes.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine, complete

# A corpus about nature. It has NO passage about the Mona Lisa.
CORPUS = [
    "Honeybees collect nectar from flowers and turn it into honey inside the hive.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Mount Everest is the highest mountain above sea level on Earth.",
]
THRESHOLD = 0.35  # below this, treat the top hit as a MISS, not an answer

QUERY = "Who painted the Mona Lisa?"
FACT = "Leonardo"


def retrieve_best(corpus, query):
    vectors = [embed(t) for t in corpus]
    qv = embed(query)
    ranked = sorted(range(len(corpus)), key=lambda i: -cosine(qv, vectors[i]))
    top = ranked[0]
    return top, cosine(qv, vectors[top])


print("STEP 1: ask a question whose answer is NOT in the corpus")
top, sim = retrieve_best(CORPUS, QUERY)
print(f"  query        : {QUERY}")
print(f"  best passage : {CORPUS[top]!r}")
print(f"  similarity   : {sim:.3f}   (threshold to trust a hit = {THRESHOLD})")
miss_detected = sim < THRESHOLD
print(f"  RETRIEVAL MISS detected (similarity below threshold)? {'yes' if miss_detected else 'no'}")

# What a naive RAG would do: ground on the irrelevant passage anyway. The answer
# cannot contain the real fact, because the fact was never retrieved.
naive = complete(f"Context: {CORPUS[top]}\nQuestion: {QUERY}")
print(f"  naive grounded answer: {naive!r}  (does NOT contain '{FACT}': "
      f"{'correct, it is a miss' if FACT not in naive else 'unexpected'})")

print("")
print("STEP 2: FIX the retrieval by adding the missing document, then retry")
fixed_corpus = CORPUS + [
    "Leonardo da Vinci painted the Mona Lisa in Florence around the year 1503.",
]
top2, sim2 = retrieve_best(fixed_corpus, QUERY)
print(f"  best passage : {fixed_corpus[top2]!r}")
print(f"  similarity   : {sim2:.3f}")
fixed_answer = complete(f"Context: {fixed_corpus[top2]}\nQuestion: {QUERY}")
print(f"  grounded answer: {fixed_answer!r}")

# The invariant: we REPRODUCED the failure (a miss with a wrong, ungrounded
# answer), then REPAIRED it (a strong hit whose answer carries the real fact).
before_bad = miss_detected and (FACT not in naive)
after_good = (sim2 >= THRESHOLD) and (FACT in fixed_answer)
ok = before_bad and after_good
print("")
print(f"FAILURE REPRODUCED THEN FIXED: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("The lesson: when a RAG answer is wrong, check retrieval FIRST. A low top")
print("similarity means the answer was never fetched. Next: measure this honestly.")
