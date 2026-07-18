#!/usr/bin/env python3
"""
LAB MM6: Capstone, a tiny multimodal assistant end to end.

Every piece of this course now snaps together into one assistant that takes an
IMAGE, an AUDIO command, and a TEXT question, and produces a grounded answer.
The flow is the real one an applied multimodal system runs:
  1. perceive the image  -> classify its visual concept (the CLIP-style encoder),
  2. perceive the audio  -> classify the spoken command (the speech features),
  3. retrieve            -> use the perceived concept to pull the right item from
                            a multimodal store (multimodal RAG),
  4. answer              -> ground a text response on the retrieved item with the
                            shared mock LLM, so nothing is invented.
This lab drives all four stages and proves each one landed: the image was read,
the command was heard, the correct item was retrieved, and the final answer is
grounded in that item. Fully deterministic and offline.

Run: python3 modules/academy-content/labs/multimodal/mm6-assistant.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine, complete

import numpy as np

IMG = 8
SR, DUR, FRAME, HOP = 800, 0.5, 80, 40


# --- perception: vision ----------------------------------------------------- #
def make_image(kind):
    a = np.zeros((IMG, IMG), dtype=np.float32)
    if kind == "vertical":   a[:, 1::2] = 1.0
    elif kind == "horizontal": a[1::2, :] = 1.0
    elif kind == "center":   a[2:6, 2:6] = 1.0
    return a


def image_concept(a):
    """Classify the image into a concept word (the CLIP-style read)."""
    col_var = float(a.mean(axis=0).var())
    row_var = float(a.mean(axis=1).var())
    center = float(a[2:6, 2:6].mean())
    edges = float(a.mean() * 2 - center)
    if center > 0.6 and center > edges:
        return "center"
    return "vertical" if col_var > row_var else "horizontal"


# --- perception: speech ----------------------------------------------------- #
def synth(freq):
    n = int(SR * DUR)
    t = np.arange(n, dtype=np.float32) / SR
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def zcr_feature(wave):
    fr = [wave[s:s + FRAME] for s in range(0, len(wave) - FRAME + 1, HOP)]
    return float(np.mean([np.mean(np.abs(np.diff(np.sign(f))) > 0) for f in fr]))


LOW_HZ, HIGH_HZ = 40.0, 160.0
_THRESH = (zcr_feature(synth(LOW_HZ)) + zcr_feature(synth(HIGH_HZ))) / 2.0


def speech_command(wave):
    """Classify the spoken command from its pitch feature: describe vs skip."""
    return "describe" if zcr_feature(wave) < _THRESH else "skip"


# --- knowledge: multimodal store -------------------------------------------- #
CONCEPT_TEXT = {
    "vertical": "vertical stripes of bright and dark columns",
    "horizontal": "horizontal stripes of bright and dark rows",
    "center": "a bright glowing block in the center of the image",
}
LIBRARY = [
    {"id": "card-A", "kind": "vertical",
     "fact": "Context: The vertical bars chart shows revenue rising each quarter. Question: what does the image show?"},
    {"id": "card-B", "kind": "horizontal",
     "fact": "Context: The horizontal rows form a project timeline of milestones. Question: what does the image show?"},
    {"id": "card-C", "kind": "center",
     "fact": "Context: The bright centered block marks the target region of interest. Question: what does the image show?"},
]


def retrieve(concept):
    """Retrieve the store item whose concept text is closest to the perceived
    image concept, in the shared embedding space (multimodal RAG)."""
    qv = embed(CONCEPT_TEXT[concept])
    scored = sorted(((cosine(qv, embed(CONCEPT_TEXT[it["kind"]])), it) for it in LIBRARY),
                    key=lambda p: p[0], reverse=True)
    return scored[0][1]


def assistant(image, wave, question):
    """The full loop: see, hear, retrieve, answer."""
    concept = image_concept(image)          # 1. vision
    command = speech_command(wave)          # 2. speech
    if command != "describe":
        return {"concept": concept, "command": command, "item": None, "answer": ""}
    item = retrieve(concept)                # 3. retrieval
    answer = complete(item["fact"])         # 4. grounded answer from the mock LLM
    return {"concept": concept, "command": command, "item": item["id"], "answer": answer}


# --- drive it --------------------------------------------------------------- #
# Input: a CENTER-pattern image, a LOW tone ("describe" command), a text question.
image = make_image("center")
wave = synth(LOW_HZ)
question = "what is in this image"

print("STEP 1: the assistant perceives, retrieves, and answers")
out = assistant(image, wave, question)
print(f"  vision  -> image concept : {out['concept']!r}")
print(f"  speech  -> spoken command: {out['command']!r}")
print(f"  rag     -> retrieved item: {out['item']!r}")
print(f"  answer  -> grounded reply: {out['answer']!r}")

print("")
print("STEP 2: check every stage landed")
# Invariants across all four modalities/stages:
vision_ok = out["concept"] == "center"
speech_ok = out["command"] == "describe"
rag_ok = out["item"] == "card-C"
# The grounded answer must come from the retrieved card's context (mentions the
# centered bright block), proving the reply is grounded, not invented.
answer_ok = "bright centered block" in out["answer"].lower() or "centered block" in out["answer"].lower()
print(f"  vision read the image      : {vision_ok}")
print(f"  speech heard the command   : {speech_ok}")
print(f"  rag retrieved the right item: {rag_ok}")
print(f"  answer grounded in the item : {answer_ok}")

ok = bool(vision_ok and speech_ok and rag_ok and answer_ok)
print("")
print(f"MULTIMODAL ASSISTANT ANSWERED ACROSS MODALITIES: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Image, audio, and text flowed through one pipeline: perceive each modality,")
print("retrieve across the shared space, and answer grounded in what was found.")
print("That is the applied multimodal engineering the job asks for. Course done.")
