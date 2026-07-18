#!/usr/bin/env python3
"""
LAB MM5: Multimodal RAG, retrieving over images AND text at once.

Plain RAG retrieves text passages. Real documents are not only text: they are
PDFs full of charts, product pages with photos, tables, and screenshots. A
multimodal knowledge base embeds EACH item from both what it shows and what it
says, then stores one combined vector per item. A text query is embedded into
that same shared space and the nearest item wins, whether the match came from
the picture, the words, or both. This lab builds a tiny library where every item
has an image (a visual pattern) and a text blurb, fuses their embeddings into one
multimodal vector, and proves a text-only query retrieves the correct item, one
whose IMAGE carries the answer the words alone would not rank first.

This is exactly the retrieval move from the RAG course, now over mixed media.
Everything is deterministic and offline: synthetic images plus the shared text
embedder, ranked with cosine similarity.

Run: python3 modules/academy-content/labs/multimodal/mm5-multimodal-rag.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine

import numpy as np

IMG = 8


def make_image(kind):
    a = np.zeros((IMG, IMG), dtype=np.float32)
    if kind == "vertical":
        a[:, 1::2] = 1.0
    elif kind == "horizontal":
        a[1::2, :] = 1.0
    elif kind == "center":
        a[2:6, 2:6] = 1.0
    return a


def encode_image(a):
    """Same shared-space image encoder as the CLIP lab: read the pattern, embed
    the concept into the text space so image and text vectors are comparable."""
    col_var = float(a.mean(axis=0).var())
    row_var = float(a.mean(axis=1).var())
    center = float(a[2:6, 2:6].mean())
    edges = float(a.mean() * 2 - center)
    if center > 0.6 and center > edges:
        concept = "a bright glowing block in the center of the image"
    elif col_var > row_var:
        concept = "vertical stripes of bright and dark columns"
    else:
        concept = "horizontal stripes of bright and dark rows"
    return np.array(embed(concept), dtype=np.float32)


def fuse(text, image_kind):
    """One multimodal vector per item: average the text embedding and the image
    embedding in the shared space. Averaging keeps both signals; a real system
    may weight or concatenate them, but the principle is identical."""
    tv = np.array(embed(text), dtype=np.float32)
    iv = encode_image(make_image(image_kind))
    v = (tv + iv) / 2.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


# A small multimodal library. Note item 2's blurb is deliberately generic ("a
# figure from the report"); its ANSWER lives in the image (a centered bright
# block), so only multimodal fusion ranks it first for a center-pattern query.
LIBRARY = [
    {"id": "doc-A", "text": "quarterly revenue chart with columns going up and down", "image": "vertical"},
    {"id": "doc-B", "text": "a timeline diagram laid out in horizontal rows", "image": "horizontal"},
    {"id": "doc-C", "text": "a figure from the report", "image": "center"},
]
store = [(item["id"], fuse(item["text"], item["image"])) for item in LIBRARY]

QUERY = "which figure shows a bright glowing block in the center"
GOLD = "doc-C"

print("STEP 1: build the multimodal store (one fused vector per item)")
for item in LIBRARY:
    print(f"  {item['id']}  text={item['text']!r:52}  image={item['image']}")

print("")
print("STEP 2: embed the text query and rank every item by cosine similarity")
qv = np.array(embed(QUERY), dtype=np.float32)
scored = sorted(((cosine(qv, v), cid) for cid, v in store), reverse=True)
for sim, cid in scored:
    print(f"  cosine={sim:+.3f}  {cid}")

best_sim, best_id = scored[0]
print("")
print(f"STEP 3: nearest multimodal item is {best_id!r}")

# Invariants for a working multimodal retriever:
#  (a) the retrieved item is the correct one (GOLD),
#  (b) it strictly beats the worst-ranked item.
ok = (best_id == GOLD) and (best_sim > scored[-1][0])
print("")
print(f"MULTIMODAL RAG RETRIEVED THE RIGHT ITEM: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("The winning item's text was vague; its IMAGE carried the answer, and")
print("fusing both put it on top. That is why multimodal RAG beats text-only")
print("over real documents. Next: wire image, audio, and text into one assistant.")
