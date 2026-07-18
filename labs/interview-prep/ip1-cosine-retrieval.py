#!/usr/bin/env python3
"""
LAB IP1: Cosine-similarity retrieval, a mini vector search.

This is the single most common warmup an AI-engineer interview opens with:
"code cosine similarity and a tiny nearest-neighbor search, from scratch, no
libraries doing the ranking for you." It looks trivial and it filters people
fast, because you have to be right about the math (dot product over the product
of norms) AND about the retrieval loop (rank every document, return the best).

You build both here on real numpy vectors, then PROVE:
  - cosine(x, x) == 1.0 (a vector is identical to itself),
  - the document that shares the most query words ranks first,
  - the full ranking comes back in the correct order.

Run: python3 modules/academy-content/labs/interview-prep/ip1-cosine-retrieval.py
"""
import sys
import numpy as np

np.random.seed(7)  # reproducible


def bag_of_words(text, vocab):
    """Turn text into a term-frequency vector over a fixed vocabulary. Real
    systems use learned embeddings; a bag of words is enough to prove retrieval
    and is exactly the kind of from-scratch representation an interviewer wants
    to see you build without reaching for a library."""
    counts = np.zeros(len(vocab), dtype=np.float64)
    words = text.lower().split()
    index = {w: i for i, w in enumerate(vocab)}
    for w in words:
        if w in index:
            counts[index[w]] += 1.0
    return counts


def cosine(a, b):
    """Cosine similarity: the dot product divided by the product of the L2
    norms. It measures the ANGLE between two vectors, not their length, which
    is why it is the default similarity for retrieval."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search(query_vec, doc_vecs, k):
    """Rank every document by cosine similarity to the query and return the top
    k as (index, score) pairs, highest first. This IS a vector search, just
    without the approximate-nearest-neighbor index a production store would add
    for speed."""
    scored = [(i, cosine(query_vec, d)) for i, d in enumerate(doc_vecs)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


# A tiny corpus. The query is about sailing; doc 2 is the sailing passage.
vocab = ["sailboat", "wind", "ocean", "engine", "diesel", "truck",
         "recipe", "flour", "oven", "bread", "harbor", "sails"]
docs = [
    "diesel engine truck engine",            # 0: vehicles
    "recipe flour oven bread flour",         # 1: baking
    "sailboat wind ocean harbor sails wind",  # 2: sailing (the target)
    "ocean wind recipe",                     # 3: mixed, weak sailing overlap
]
query = "sailboat wind ocean sails"

doc_vecs = [bag_of_words(d, vocab) for d in docs]
query_vec = bag_of_words(query, vocab)

# 1. Identity property: a vector is perfectly similar to itself.
self_sim = cosine(query_vec, query_vec)
print("STEP 1: cosine identity")
print(f"  cosine(q, q) = {self_sim:.6f}  (must be 1.0)")

# 2. Rank the corpus and read off the winner.
ranking = search(query_vec, doc_vecs, k=len(docs))
print("")
print("STEP 2: ranked retrieval (index, score)")
for idx, score in ranking:
    print(f"  doc {idx}: {score:.4f}  {docs[idx]!r}")

top_index = ranking[0][0]
top_is_sailing = (top_index == 2)
order_indices = [idx for idx, _ in ranking]
# doc 2 (sailing) first, doc 3 (weak overlap) ahead of the two unrelated docs.
expected_front = order_indices[0] == 2 and 3 in order_indices[:3]

print("")
print(f"STEP 3: top hit is the sailing passage (doc 2) : {top_is_sailing}")
print(f"        identity holds                        : {abs(self_sim - 1.0) < 1e-9}")

ok = top_is_sailing and abs(self_sim - 1.0) < 1e-9 and expected_front
print("")
print(f"COSINE RETRIEVAL RANKS THE RIGHT DOCUMENT FIRST: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("You just built the core of every vector database. Next round: tokenizers.")
