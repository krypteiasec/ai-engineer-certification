#!/usr/bin/env python3
"""
LAB MM2: CLIP-style matching, one shared space for images and text.

The idea that unlocked multimodal AI is simple to say and powerful in practice:
put images and text into the SAME vector space, so a picture and the words that
describe it land close together. Then "which caption fits this image" becomes
"which caption vector is closest", the exact cosine-similarity move you learned
for text retrieval, now working ACROSS modalities. This lab builds three tiny
synthetic images with clearly different visual patterns, a deterministic image
encoder that reads each image's pattern and projects it into the shared text
space, and then matches every image to its correct caption. It proves the right
image-caption pair scores higher than any wrong pair.

The image encoder here detects the dominant visual pattern from real pixel
statistics and embeds that concept with the shared text embedder. That stands in
for CLIP's LEARNED image projection, but the matching it produces is real: it is
measured with cosine similarity, and the correct pairing wins on the numbers.

Run: python3 modules/academy-content/labs/multimodal/mm2-clip-matching.py
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
    """Build a tiny 8x8 grayscale image with one clear visual pattern."""
    a = np.zeros((IMG, IMG), dtype=np.float32)
    if kind == "vertical":
        a[:, 1::2] = 1.0            # bright every other COLUMN -> vertical stripes
    elif kind == "horizontal":
        a[1::2, :] = 1.0            # bright every other ROW -> horizontal stripes
    elif kind == "center":
        a[2:6, 2:6] = 1.0          # a bright block in the CENTER
    return a


def encode_image(a):
    """Deterministic image encoder: read the pixels, decide the dominant visual
    concept, and project it into the SHARED text-embedding space via embed().

    We measure how much brightness varies between columns vs between rows, and
    how bright the center is. Those real statistics pick the concept word, which
    is what a learned CLIP image head does implicitly. The output vector lives in
    the same 64-dim space as any text embedding, so cosine can compare them."""
    col_means = a.mean(axis=0)      # average brightness of each column
    row_means = a.mean(axis=1)      # average brightness of each row
    col_var = float(col_means.var())
    row_var = float(row_means.var())
    center = float(a[2:6, 2:6].mean())
    edges = float(a.mean() * 2 - center)  # rough non-center brightness

    if center > 0.6 and center > edges:
        concept = "a bright glowing block in the center of the image"
    elif col_var > row_var:
        concept = "vertical stripes of bright and dark columns"
    else:
        concept = "horizontal stripes of bright and dark rows"
    return embed(concept)


# Three images and their true captions. The captions are written the way a person
# would describe each picture, in their own words.
KINDS = ["vertical", "horizontal", "center"]
CAPTIONS = [
    "a photo of vertical stripes running up and down in columns",
    "a photo of horizontal stripes running across in rows",
    "a bright glowing square right in the center of the frame",
]

images = [make_image(k) for k in KINDS]
img_vecs = [encode_image(im) for im in images]
cap_vecs = [embed(c) for c in CAPTIONS]

print("STEP 1: encode every image and every caption into the shared space")
for i, k in enumerate(KINDS):
    print(f"  image[{i}] pattern={k:<10} -> a {len(img_vecs[i])}-dim shared vector")

print("")
print("STEP 2: score every image against every caption with cosine similarity")
print("        (rows = images, columns = captions; the diagonal is the truth)")
S = np.zeros((len(images), len(CAPTIONS)), dtype=np.float32)
for i in range(len(images)):
    for j in range(len(CAPTIONS)):
        S[i, j] = cosine(img_vecs[i], cap_vecs[j])
header = "            " + "".join(f"cap{j:<7}" for j in range(len(CAPTIONS)))
print(header)
for i in range(len(images)):
    row = "  ".join(f"{S[i, j]:+.3f}" for j in range(len(CAPTIONS)))
    star = "  <- image %d" % i
    print(f"  image{i}    {row}{star}")

print("")
print("STEP 3: match each image to its best caption")
# For every image, the correct caption is the one with the highest similarity,
# and it must beat every wrong caption for that image (a strict argmax on the
# diagonal). We also confirm each caption's best image is the matching one.
row_ok = all(int(np.argmax(S[i])) == i for i in range(len(images)))
col_ok = all(int(np.argmax(S[:, j])) == j for j in range(len(CAPTIONS)))
for i in range(len(images)):
    picked = int(np.argmax(S[i]))
    print(f"  image{i} best caption = cap{picked}  ({'correct' if picked == i else 'WRONG'})")

ok = bool(row_ok and col_ok)
print("")
print(f"CLIP-STYLE MATCHER PAIRED IMAGE TO CAPTION: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("One shared space turned 'describe this image' into 'find the nearest")
print("caption vector'. That is the engine behind image search, zero-shot")
print("classification, and captioning. Next: generating an image, not reading one.")
