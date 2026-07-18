#!/usr/bin/env python3
"""
LAB 01: Build a tokenizer from scratch (character level).

This is the first thing every language model needs: a way to turn text into
numbers and numbers back into text. No libraries doing it for you, no magic.
You are building the exact component GPT has, at the simplest possible level:
one token per character. Then you hand the ids to PyTorch as a tensor, because
that is the shape every model in this course actually consumes.

Run:   python3 modules/academy-content/labs/lab-01-tokenizer.py

It prints the vocabulary it learned, encodes a sentence into integers, wraps
them in a torch tensor, decodes back to text, and PROVES the round-trip is
lossless (decode(encode(text)) == text). If that assertion ever fails, the
tokenizer is broken: that check is your safety net for the rest of the course.
"""
import sys
import torch

# The corpus. In a real LLM this is billions of characters of text. Here it is
# one small paragraph: enough to build a real vocabulary from.
CORPUS = """the raven flew over the quiet sea.
scout the dog watched the sky turn gold.
every token is a number the model can learn."""


# Step 1: BUILD THE VOCABULARY.
# A vocabulary is just the sorted set of every unique character in the data.
# Sorting makes it deterministic: same corpus -> same vocab -> same ids, always.
def build_vocab(text):
    return sorted(set(text))  # set() dedupes, sorted() gives a stable order


# Step 2: BUILD THE TWO LOOKUP MAPS.
# stoi = string -> integer (encode).   itos = integer -> string (decode).
# The token id of a character is simply its position in the sorted vocab.
def build_maps(vocab):
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return stoi, itos


# Step 3: ENCODE.  text -> list of token ids.
def encode(text, stoi):
    ids = []
    for ch in text:
        if ch not in stoi:
            raise ValueError("unknown character: %r" % ch)
        ids.append(stoi[ch])
    return ids


# Step 4: DECODE.  list of token ids -> text.
def decode(ids, itos):
    out = []
    for i in ids:
        i = int(i)  # ids may arrive as tensor scalars
        if i not in itos:
            raise ValueError("unknown token id: %d" % i)
        out.append(itos[i])
    return "".join(out)


# Now run it, step by step. (Kept at top level, not wrapped in main(), so each
# step is its own notebook cell you can run and inspect on its own.)
vocab = build_vocab(CORPUS)
stoi, itos = build_maps(vocab)

print("STEP 1: Vocabulary learned from the corpus")
print("  size: %d unique tokens" % len(vocab))
shown = " ".join("\\n" if c == "\n" else "_" if c == " " else c for c in vocab)
print("  tokens: %s" % shown)

sentence = "the raven watched the sea."
ids = encode(sentence, stoi)
# This is the real handoff: the model consumes a torch tensor of ids, not a
# Python list. torch.long is the integer dtype embeddings expect.
ids_tensor = torch.tensor(ids, dtype=torch.long)
print("STEP 2: Encode: text becomes a torch tensor of ids")
print("  text : %r" % sentence)
print("  tensor: %s  (dtype=%s, shape=%s)" % (ids_tensor.tolist(), ids_tensor.dtype, tuple(ids_tensor.shape)))

back = decode(ids_tensor, itos)
print("STEP 3: Decode: numbers become text again")
print("  ids  : %s" % ids_tensor.tolist())
print("  text : %r" % back)

# Step 5: THE INVARIANT. A tokenizer that loses information is useless.
# decode(encode(x)) must equal x for ANY text over the vocabulary.
# We prove it here on the whole corpus, character for character.
round_trip = decode(encode(CORPUS, stoi), itos)
lossless = round_trip == CORPUS
print("STEP 4: The invariant: decode(encode(text)) == text")
print("  lossless round-trip over full corpus: %s" % ("YES" if lossless else "NO"))
if not lossless:
    print("  FAILED: the tokenizer lost information. Fix it before moving on.", file=sys.stderr)
    sys.exit(1)
print("")
print("Done. You built a working tokenizer. Every model in this course starts here.")
