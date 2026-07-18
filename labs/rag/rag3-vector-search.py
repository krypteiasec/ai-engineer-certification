#!/usr/bin/env python3
"""
LAB RAG3: Vector search, nearest neighbors.

Retrieval is nearest-neighbor search: embed every document once, embed the query,
and return the k documents whose vectors are closest. This lab builds a small but
REAL vector store: an index of (id, vector) pairs and a search(query, k) function
that returns the top-k by cosine. This is exactly what a production vector
database (FAISS, pgvector, Pinecone) does, just without the speed tricks. You
prove the top result is correct and that the true answer is inside the top-k.

Run: python3 modules/academy-content/labs/rag/rag3-vector-search.py
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
    "The Eiffel Tower in Paris was completed in 1889 for the World Fair.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Mount Everest is the highest mountain above sea level on Earth.",
]


class VectorStore:
    """The smallest honest vector store: embed once at index time, then rank by
    cosine at query time. index() is the write path, search() is the read path."""

    def __init__(self):
        self.ids = []
        self.vectors = []
        self.texts = []

    def index(self, doc_id, text):
        self.ids.append(doc_id)
        self.vectors.append(embed(text))  # embed ONCE, store the vector
        self.texts.append(text)

    def search(self, query, k=3):
        qv = embed(query)
        scored = [(cosine(qv, self.vectors[i]), self.ids[i], self.texts[i])
                  for i in range(len(self.ids))]
        scored.sort(reverse=True)          # highest cosine first
        return scored[:k]                  # top-k nearest neighbors


store = VectorStore()
for i, text in enumerate(CORPUS):
    store.index(i, text)
print(f"STEP 1: indexed {len(CORPUS)} documents as vectors")

QUERY = "what is the deepest ocean on earth"
GOLD = 3  # the Pacific Ocean passage
print("")
print(f"STEP 2: search(query, k=3) for {QUERY!r}")
hits = store.search(QUERY, k=3)
for rank, (sim, doc_id, text) in enumerate(hits, start=1):
    print(f"  #{rank}  cosine={sim:+.3f}  [{doc_id}] {text}")

top_id = hits[0][1]
retrieved_ids = [doc_id for _, doc_id, _ in hits]

# Invariants a retriever must satisfy:
#  (a) the top-1 nearest neighbor is the correct document,
#  (b) the correct document is present in the returned top-k (recall@k = 1).
ok = (top_id == GOLD) and (GOLD in retrieved_ids)
print("")
print(f"TOP-1 RETRIEVED THE CORRECT PASSAGE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("You have a working retriever. Next: documents are long, so we must split")
print("them into chunks small enough to embed and retrieve precisely.")
