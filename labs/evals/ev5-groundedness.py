#!/usr/bin/env python3
"""
LAB EV5: Groundedness and hallucination scoring.

A confident wrong answer is the most dangerous failure an LLM system has. The
defense is a GROUNDEDNESS check: every claim in the answer must be supported by
the provided context. In this lab you take a grounded answer (produced from the
context) and a hallucinated answer (facts that are NOT in the context), and score
each with a faithfulness grader: the fraction of the answer's content words that
appear in the source. Grounded answers score high, hallucinations score low, and
the grader flags the hallucination before it ever reaches a user. A cosine check
backs it up as a second, cheaper signal.

Run: python3 modules/academy-content/labs/evals/ev5-groundedness.py
"""
import sys, os, re
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

CONTEXT = ("The Apollo 11 mission landed on the moon in 1969. "
           "Neil Armstrong was the first person to walk on it.")

# A GROUNDED answer, produced by answering strictly from the context.
grounded = complete(f"Context: {CONTEXT}\nQuestion: Who was the first person to walk on the moon?")

# A HALLUCINATED answer: fluent, confident, and not supported by the context.
hallucinated = "The mission landed on Mars in 1972."

STOP = {"the", "a", "an", "on", "in", "of", "to", "is", "was", "it", "and", "for"}

def content_words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

# THE FAITHFULNESS GRADER: fraction of the answer's content words found in the
# source. 1.0 means every claim is supported; low means the answer invented facts.
def faithfulness(answer, context):
    ctx = set(content_words(context))
    words = content_words(answer)
    if not words:
        return 0.0
    return sum(1 for w in words if w in ctx) / len(words)

THRESHOLD = 0.8
g_score = faithfulness(grounded, CONTEXT)
h_score = faithfulness(hallucinated, CONTEXT)
g_cos = cosine(embed(grounded), embed(CONTEXT))
h_cos = cosine(embed(hallucinated), embed(CONTEXT))

print("STEP 1: score a grounded answer")
print(f"  answer       : {grounded!r}")
print(f"  faithfulness : {g_score:.2f}  (cosine {g_cos:.2f})  grounded={g_score >= THRESHOLD}")
print("")
print("STEP 2: score a hallucinated answer with the same grader")
print(f"  answer       : {hallucinated!r}")
print(f"  faithfulness : {h_score:.2f}  (cosine {h_cos:.2f})  grounded={h_score >= THRESHOLD}")

# The grader must PASS the grounded answer and FLAG the hallucinated one.
flagged = h_score < THRESHOLD
passed = g_score >= THRESHOLD
ok = passed and flagged and (g_score > h_score)
print("")
print(f"FAITHFULNESS GRADER CAUGHT THE HALLUCINATION: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Ground every claim in the source. Next: pass@k and reasoning about variance.")
