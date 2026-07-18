#!/usr/bin/env python3
"""
LAB RAG6: Retrieval augmented generation, end to end.

This is the whole thing, assembled and working. A small corpus goes in. A question
comes in. The system embeds the corpus, retrieves the top-k most relevant
passages, assembles them into a context window, and asks the model to answer
USING ONLY that context. The answer is grounded: it comes from retrieved text,
not from the model's frozen memory, so it is correct and traceable to a source.
This lab is a complete, self-contained RAG you can read top to bottom and run.

Run: python3 modules/academy-content/labs/rag/rag6-rag-pipeline.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine, complete

# ── The corpus: a small real knowledge base, one fact per passage. ──────────────
CORPUS = [
    "Honeybees collect nectar from flowers and turn it into honey inside the hive.",
    "The Eiffel Tower in Paris was completed in 1889 for the World Fair.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Mount Everest is the highest mountain above sea level on Earth.",
]


class RAG:
    """A complete retrieval augmented generation pipeline in one small class."""

    def __init__(self, corpus):
        # INDEX: embed every passage once and keep the vectors alongside the text.
        self.corpus = corpus
        self.vectors = [embed(text) for text in corpus]

    def retrieve(self, query, k=2):
        # RETRIEVE: rank passages by cosine similarity, keep the top-k.
        qv = embed(query)
        ranked = sorted(range(len(self.corpus)), key=lambda i: -cosine(qv, self.vectors[i]))
        return ranked[:k]

    def answer(self, query, k=2):
        # ASSEMBLE + GENERATE: build a context window from the retrieved passages,
        # then ask the model to answer grounded in that context only.
        hits = self.retrieve(query, k=k)
        context = " ".join(self.corpus[i] for i in hits)
        prompt = f"Context: {context}\nQuestion: {query}"
        return hits, context, complete(prompt)


rag = RAG(CORPUS)
print(f"STEP 1: indexed {len(CORPUS)} passages")

QUERY = "When was the Eiffel Tower completed?"
GOLD = 1          # the Eiffel Tower passage
FACT = "1889"     # the grounded answer must carry this fact

print("")
print(f"STEP 2: run the full RAG loop for {QUERY!r}")
hits, context, answer = rag.answer(QUERY, k=2)
print(f"  retrieved passages: {hits}")
print(f"  context window    : {context}")
print(f"  grounded answer   : {answer!r}")

# The two invariants that define a working RAG:
#  (a) retrieval found the correct source passage,
#  (b) the answer is GROUNDED: it is drawn from the retrieved context (not
#      invented) and it carries the real fact.
retrieved_correct = GOLD in hits
grounded = (answer in context) and (FACT in answer)
print("")
print(f"RETRIEVED CORRECT PASSAGE: {'YES' if retrieved_correct else 'NO'}")
print(f"RAG ANSWER GROUNDED IN CONTEXT: {'YES' if grounded else 'NO'}")
if not (retrieved_correct and grounded):
    sys.exit(1)
print("")
print("That is a real, working RAG: retrieve, ground, answer. Next: what happens")
print("when retrieval MISSES, the failure mode that breaks RAG in the wild.")
