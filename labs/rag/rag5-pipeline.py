#!/usr/bin/env python3
"""
LAB RAG5: The retrieval pipeline, with a re-ranker.

A good pipeline is two stages. Stage one, RECALL: cast a wide net with fast
vector search and pull the top few candidates. Stage two, PRECISION: re-rank
those few with a sharper signal to float the truly relevant one to the top.
Vector similarity is fast but coarse, so it sometimes ranks a passage that merely
"vibes" close above the passage that actually answers the question. The re-ranker
fixes that by fusing semantic similarity with exact term overlap. This lab shows
a case where vector search alone gets the top wrong, and the re-ranker promotes
the correct passage to number one.

Run: python3 modules/academy-content/labs/rag/rag5-pipeline.py
"""
import sys, os, re
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine

CORPUS = [
    "Honeybees collect nectar from flowers and turn it into honey inside the hive.",
    "The Eiffel Tower in Paris was completed in 1889 for the World Fair.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "A honey badger is a fearless mammal with thick loose skin and no fear.",
]
VECTORS = [embed(text) for text in CORPUS]  # index once
QUERY = "which animals collect nectar"
GOLD = 0  # the honeybees passage is the correct answer


def _terms(text):
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def recall_stage(query, k=4):
    """Fast, coarse: return the top-k candidate ids by cosine similarity."""
    qv = embed(query)
    scored = sorted(range(len(CORPUS)), key=lambda i: -cosine(qv, VECTORS[i]))
    return scored[:k]


def rerank_stage(query, candidate_ids):
    """Sharper: fuse semantic similarity with exact query-term overlap, then sort.
    Overlap is a strong precision signal a coarse vector score can miss."""
    qv = embed(query)
    q_terms = _terms(query)
    def fused(i):
        sem = cosine(qv, VECTORS[i])
        lexical = len(q_terms & _terms(CORPUS[i]))
        return sem + 0.15 * lexical
    return sorted(candidate_ids, key=lambda i: -fused(i))


print(f"STEP 1: RECALL stage, vector search top-4 for {QUERY!r}")
candidates = recall_stage(QUERY, k=4)
for i in candidates:
    print(f"  cand [{i}] cosine={cosine(embed(QUERY), VECTORS[i]):+.3f}  {CORPUS[i]}")
vector_top = candidates[0]
print(f"  vector-search top-1 = [{vector_top}] {CORPUS[vector_top]}")

print("")
print("STEP 2: PRECISION stage, re-rank the candidates by fused score")
reranked = rerank_stage(QUERY, candidates)
for rank, i in enumerate(reranked, start=1):
    print(f"  #{rank}  [{i}] {CORPUS[i]}")
reranked_top = reranked[0]
print(f"  re-ranked top-1 = [{reranked_top}] {CORPUS[reranked_top]}")

# The invariant that proves the re-ranker earned its keep: vector search alone put
# the WRONG passage first, and the re-ranker moved the CORRECT one to the top.
promoted = (vector_top != GOLD) and (reranked_top == GOLD)
print("")
print(f"  vector search alone was correct? {'yes' if vector_top == GOLD else 'no'}")
print(f"RE-RANKER PROMOTED THE CORRECT PASSAGE TO TOP: {'YES' if promoted else 'NO'}")
if not promoted:
    sys.exit(1)
print("")
print("Recall wide, then re-rank precise. Next: feed the retrieved context to the")
print("model and generate a grounded answer, the full RAG loop.")
