#!/usr/bin/env python3
"""
LAB MM1: An image is numbers, and a vision model reads it in patches.

A model cannot see a picture. What it gets is a grid of numbers: the brightness
of each pixel. A vision transformer does not read those pixels one at a time,
it cuts the image into small square PATCHES, flattens each patch into a short
vector, and projects every patch into an embedding, exactly the way a text model
turns each token into an embedding. From there a patch is just a token. This lab
builds a tiny 8x8 grayscale "image" by hand, cuts it into 2x2 patches, projects
each patch into an embedding, and proves every shape along the way is what the
transformer expects. No network, no real image, just the mechanics made visible.

Run: python3 modules/academy-content/labs/multimodal/mm1-image-patches.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

import numpy as np

IMG = 8          # the image is IMG x IMG pixels
PATCH = 2        # each patch is PATCH x PATCH pixels
EMBED_DIM = 16   # each patch becomes a vector of this many numbers

print("STEP 1: build a tiny image as a grid of pixel brightnesses")
# A deterministic 8x8 pattern: vertical stripes (even columns dark, odd bright).
# Real images are just bigger grids of the same kind of numbers.
img = np.zeros((IMG, IMG), dtype=np.float32)
img[:, 1::2] = 1.0
print(f"  image shape     : {img.shape}  (a {IMG}x{IMG} grid of pixels)")
print(f"  pixel range     : min={img.min():.0f} max={img.max():.0f}")

print("")
print("STEP 2: cut the image into non-overlapping PATCHES")
# Slice the grid into PATCH x PATCH blocks, then flatten each block into a row.
n_side = IMG // PATCH               # patches per side
n_patches = n_side * n_side         # total patches
patch_len = PATCH * PATCH           # pixels per patch (the flattened length)
patches = np.zeros((n_patches, patch_len), dtype=np.float32)
p = 0
for r in range(0, IMG, PATCH):
    for c in range(0, IMG, PATCH):
        block = img[r:r+PATCH, c:c+PATCH]     # a small PATCH x PATCH square
        patches[p] = block.reshape(-1)        # flatten it into one row vector
        p += 1
print(f"  patch grid      : {n_side}x{n_side} = {n_patches} patches")
print(f"  patches shape   : {patches.shape}  ({n_patches} patches, each {patch_len} pixels)")

print("")
print("STEP 3: project every patch into an embedding (a linear layer)")
# A fixed deterministic projection matrix stands in for the learned one.
# patches (n_patches, patch_len) @ W (patch_len, EMBED_DIM) -> (n_patches, EMBED_DIM).
rng = np.random.default_rng(0)
W = rng.standard_normal((patch_len, EMBED_DIM)).astype(np.float32)
patch_embeddings = patches @ W
print(f"  projection W    : {W.shape}")
print(f"  embeddings shape: {patch_embeddings.shape}  (one {EMBED_DIM}-vector per patch)")

# Invariants a real vision transformer relies on:
#  (a) the number of patches is (IMG/PATCH)^2,
#  (b) each patch flattens to PATCH*PATCH pixels,
#  (c) each patch becomes exactly one EMBED_DIM embedding,
#  (d) no pixels are lost: patches hold the same values as the image.
shapes_ok = (
    patches.shape == (n_patches, patch_len)
    and patch_embeddings.shape == (n_patches, EMBED_DIM)
    and n_patches == (IMG // PATCH) ** 2
)
pixels_preserved = np.isclose(patches.sum(), img.sum())
ok = bool(shapes_ok and pixels_preserved)

print("")
print(f"IMAGE TO PATCHES TO PATCH EMBEDDINGS, SHAPES CHECK: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("A picture is a grid of numbers, and a vision model reads it as a short")
print("sequence of patch embeddings. From here a patch behaves exactly like a")
print("token. Next: put image and text into ONE shared space so they can match.")
