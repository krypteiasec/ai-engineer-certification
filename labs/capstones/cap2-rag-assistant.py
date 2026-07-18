#!/usr/bin/env python3
"""
CAPSTONE 2 (ch2): A complete RAG assistant that answers with citations.

This is Capstone project one, the deployed RAG assistant, built end to end and
slightly larger than the RAG course lab. It indexes a small real corpus, answers
a batch of questions grounded ONLY in retrieved passages, and CITES the source of
every answer. Citations are the difference between a demo and something a manager
trusts: an answer you can trace to a source is defensible, an answer you cannot is
a liability.

The assistant integrates the patterns from the RAG course (embed, cosine, top-k
retrieval, grounded generation) and adds two production touches: every answer
carries the index of the passage it came from (the citation), and a question with
no good match returns a SAFE FALLBACK ("I do not have that in my sources") instead
of hallucinating. It answers three in-corpus questions correctly with citations
and refuses one out-of-corpus question, then prints the success invariant.

Run: python3 modules/academy-content/labs/capstones/cap2-rag-assistant.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine, complete
import numpy as np
import re

# Common words carry no topic, so they do not count as a real retrieval match.
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "on", "in", "to",
             "for", "and", "or", "what", "when", "where", "which", "how", "that",
             "this", "it", "its", "at", "by", "with", "as", "be", "do", "does"}


def _content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}

# ── The corpus: a small real knowledge base, one fact per passage. ───────────────
CORPUS = [
    "The Eiffel Tower in Paris was completed in 1889 for the World Fair.",
    "Honeybees collect nectar from flowers and turn it into honey inside the hive.",
    "The Pacific Ocean is the largest and deepest ocean on planet Earth.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
]
FALLBACK = "I do not have that in my sources."


class RagAssistant:
    """A deployed-shaped RAG assistant: index once, retrieve, answer, CITE, and
    refuse safely when retrieval finds nothing relevant."""

    def __init__(self, corpus):
        self.corpus = corpus
        # Index: embed every passage once, store as a numpy matrix for fast search.
        self.matrix = np.array([embed(text) for text in corpus], dtype=float)

    def retrieve(self, query, k=2):
        qv = np.array(embed(query), dtype=float)
        sims = self.matrix @ qv  # vectors are unit-length, so dot == cosine
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order[:k]]

    def answer(self, query, k=2):
        hits = self.retrieve(query, k=k)
        best_i, _ = hits[0]
        # Safe fallback: refuse unless the top passage genuinely shares a topic
        # word with the question. This is the honest grounding check: a question
        # about a subject the corpus never mentions has zero content-word overlap
        # with its nearest passage, so we refuse instead of inventing an answer.
        if not (_content_words(query) & _content_words(self.corpus[best_i])):
            return {"answer": FALLBACK, "citations": [], "grounded": True}
        context = " ".join(self.corpus[i] for i, _ in hits)
        text = complete("Context: %s\nQuestion: %s" % (context, query))
        # A citation is only valid if the cited passage actually contains the answer.
        cites = [i for i, _ in hits if text and text in self.corpus[i]]
        grounded = bool(text) and text in context
        return {"answer": text, "citations": cites, "grounded": grounded}


def main():
    bot = RagAssistant(CORPUS)
    print("STEP 1: indexed %d passages into the assistant\n" % len(CORPUS))

    # (question, must-appear fact, expected citation index)
    QUESTIONS = [
        ("When was the Eiffel Tower completed?", "1889", 0),
        ("What is the largest ocean on Earth?", "Pacific", 2),
        ("What is the highest mountain on Earth?", "Everest", 3),
    ]
    print("STEP 2: answer in-corpus questions WITH citations")
    answered_ok = 0
    for q, fact, want_cite in QUESTIONS:
        r = bot.answer(q)
        cited = want_cite in r["citations"]
        carries_fact = fact.lower() in r["answer"].lower()
        good = r["grounded"] and cited and carries_fact
        answered_ok += 1 if good else 0
        print("  Q: %-42s cite=%s fact=%s -> %s" % (
            q, r["citations"], "YES" if carries_fact else "no",
            "OK" if good else "FAIL"))

    print("\nSTEP 3: an out-of-corpus question must REFUSE, not hallucinate")
    oob = bot.answer("What is the boiling point of mercury?")
    refused = oob["answer"] == FALLBACK and oob["citations"] == []
    print("  Q: What is the boiling point of mercury? -> %r (refused=%s)" % (
        oob["answer"], refused))

    all_cited = answered_ok == len(QUESTIONS)
    ok = all_cited and refused
    print("")
    print("  answered %d/%d grounded + cited, refused the unknown: %s" % (
        answered_ok, len(QUESTIONS), "YES" if refused else "NO"))
    print("")
    print("RAG ASSISTANT ANSWERED WITH CITATIONS: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Deployed-shaped: retrieves, grounds, cites, and refuses safely. Capstone one.")


if __name__ == "__main__":
    main()
