#!/usr/bin/env python3
"""
LAB RAG4: Chunking documents well.

You cannot embed a whole book as one vector, the meaning smears out and retrieval
gets vague. So you split each document into CHUNKS: pieces small enough to have a
single clear meaning, big enough to stand alone as an answer. The two dials are
chunk size and overlap. Overlap matters because a fact can straddle a boundary,
and a little repetition keeps it whole in at least one chunk. This lab builds a
sliding-window chunker, proves it loses nothing (every word is covered), and
proves the chunk holding a buried fact is the one retrieval returns.

Run: python3 modules/academy-content/labs/rag/rag4-chunking.py
"""
import sys, os, re
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine

# One long document. The answer we will search for is buried in the middle.
DOCUMENT = (
    "Honeybees live together in a colony inside a wax hive. Worker bees fly out "
    "to flowers and collect nectar in a special stomach. Back at the hive they "
    "pass the nectar mouth to mouth and fan it with their wings. A single hive "
    "can hold about fifty thousand bees at the height of summer. The queen bee "
    "can lay up to two thousand eggs in a single day. Guard bees stand at the "
    "entrance and check every returning forager for the colony scent."
)


def chunk(text, size=14, overlap=4):
    """Sliding window over WORDS. `size` words per chunk, stepping forward by
    (size - overlap) so neighboring chunks share `overlap` words. Overlap is what
    keeps a fact whole even when it lands near a boundary."""
    words = text.split()
    step = size - overlap
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += step
    return chunks


chunks = chunk(DOCUMENT)
print(f"STEP 1: split the document into {len(chunks)} overlapping chunks")
for i, c in enumerate(chunks):
    print(f"  [{i}] {c}")

# STEP 2: coverage invariant. Chunking must LOSE NOTHING: every word of the
# original document has to appear in at least one chunk.
doc_words = DOCUMENT.split()
covered = set()
for c in chunks:
    for w in c.split():
        covered.add(w)
every_word_covered = all(w in covered for w in doc_words)
print("")
print(f"STEP 2: every word covered by some chunk? {'yes' if every_word_covered else 'no'}")

# STEP 3: retrieval over chunks. The fact "two thousand eggs" is buried mid-doc.
QUERY = "how many eggs does the queen lay in a day"
ANSWER_WORD = "eggs"
qv = embed(QUERY)
ranked = sorted(range(len(chunks)), key=lambda i: -cosine(qv, embed(chunks[i])))
best = ranked[0]
print("")
print(f"STEP 3: query {QUERY!r}")
print(f"  best chunk [{best}]: {chunks[best]}")

# Invariant: chunking is lossless AND the chunk that holds the answer is the one
# retrieval returns for the matching query.
ok = every_word_covered and (ANSWER_WORD in chunks[best].lower())
print("")
print(f"CHUNK CONTAINING THE ANSWER WAS RETRIEVED: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Good chunks make retrieval precise. Next: wire embed, store, search, and a")
print("re-ranker into one pipeline.")
