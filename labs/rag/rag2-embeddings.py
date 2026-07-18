#!/usr/bin/env python3
"""
LAB RAG2: Embeddings for retrieval.

An embedding turns a piece of text into a vector: a fixed list of numbers that
captures its meaning. The magic property is that text about the same thing lands
in a SIMILAR direction, so two passages about honey point one way and a passage
about oceans points another. Once meaning is a direction, "find the relevant
passage" becomes "find the closest vector", which is just arithmetic. This lab
embeds a small corpus and a query, measures similarity with cosine, and proves
the nearest vector really is the passage a human would pick.

Run: python3 modules/academy-content/labs/rag/rag2-embeddings.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine

CORPUS = [
    "Honeybees collect nectar from flowers and turn it into honey inside the hive.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
]
QUERY = "how do bees make honey from flowers"
GOLD = 0  # the honey passage is the one a human would retrieve

print("STEP 1: an embedding is a fixed-length vector of meaning")
qv = embed(QUERY)
print(f"  query           : {QUERY!r}")
print(f"  embedding length: {len(qv)} numbers (always the same size)")

print("")
print("STEP 2: cosine similarity ranks every passage against the query")
scored = []
for i, passage in enumerate(CORPUS):
    sim = cosine(qv, embed(passage))
    scored.append((sim, i, passage))
    print(f"  cosine={sim:+.3f}  [{i}] {passage}")

# Rank high to low. The most similar passage is the retrieval result.
scored.sort(reverse=True)
best_sim, best_i, best_passage = scored[0]
print("")
print(f"STEP 3: nearest passage is [{best_i}] {best_passage!r}")

# Invariants that make embeddings usable for retrieval:
#  (a) the nearest vector is the passage we actually wanted (GOLD),
#  (b) the relevant passage is strictly closer than the least relevant one.
ok = (best_i == GOLD) and (best_sim > scored[-1][0])
print("")
print(f"NEAREST PASSAGE BY EMBEDDING IS THE CORRECT ONE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Meaning became a direction, and 'relevant' became 'closest'. Next: turn")
print("this into a real top-k nearest-neighbor search over a whole corpus.")
