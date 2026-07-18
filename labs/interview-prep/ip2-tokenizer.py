#!/usr/bin/env python3
"""
LAB IP2: A byte-pair-encoding (BPE) tokenizer from scratch.

"Write a tokenizer" is a classic practical-coding round because it is small
enough to finish in the time box but real enough to expose whether you actually
understand how models see text. BPE is the algorithm behind GPT and most modern
tokenizers: start from individual characters, then repeatedly merge the single
most frequent adjacent pair into a new token. Frequent sequences like "th" and
"the" become single tokens, which is why subword tokenization beats characters
at scale.

You implement training (learn merges) and greedy encoding here, then PROVE:
  - encoding then decoding round-trips back to the exact original text,
  - learning merges strictly shrinks the token count versus raw characters,
  - the expected high-frequency merge is actually learned.

Run: python3 modules/academy-content/labs/interview-prep/ip2-tokenizer.py
"""
import sys
from collections import Counter


def get_pairs(tokens):
    """Count every adjacent token pair in the sequence."""
    counts = Counter()
    for a, b in zip(tokens, tokens[1:]):
        counts[(a, b)] += 1
    return counts


def train_bpe(text, num_merges):
    """Learn a list of merges. Each step finds the most frequent adjacent pair
    across the corpus and records it. Returns the ordered merge list, which is
    the tokenizer's learned vocabulary beyond the base characters."""
    tokens = list(text)  # base vocabulary is the set of characters
    merges = []
    for _ in range(num_merges):
        pairs = get_pairs(tokens)
        if not pairs:
            break
        # Most frequent pair wins; ties break deterministically on the pair text.
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if best[1] < 2:
            break  # nothing repeats, no more useful merges
        pair = best[0]
        merges.append(pair)
        tokens = apply_merge(tokens, pair)
    return merges


def apply_merge(tokens, pair):
    """Replace every occurrence of the adjacent `pair` with its joined token."""
    merged = []
    i = 0
    joined = pair[0] + pair[1]
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(joined)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def encode(text, merges):
    """Greedy encode: start from characters, then apply every learned merge in
    the order it was learned. This is exactly how a real BPE encoder tokenizes
    new text."""
    tokens = list(text)
    for pair in merges:
        tokens = apply_merge(tokens, pair)
    return tokens


def decode(tokens):
    """Decode is just concatenation: BPE tokens are substrings of the text."""
    return "".join(tokens)


corpus = "the theme of the theater is the thesis theme " * 4
merges = train_bpe(corpus, num_merges=8)

sample = "the theme"
char_tokens = list(sample)
bpe_tokens = encode(sample, merges)
restored = decode(bpe_tokens)

print("STEP 1: learned merges (in order)")
for a, b in merges:
    print(f"  merge {a!r} + {b!r} -> {a + b!r}")

print("")
print("STEP 2: encode a sample")
print(f"  text        : {sample!r}")
print(f"  chars ({len(char_tokens):2d}) : {char_tokens}")
print(f"  bpe   ({len(bpe_tokens):2d}) : {bpe_tokens}")
print(f"  decoded     : {restored!r}")

round_trips = (restored == sample)
compresses = (len(bpe_tokens) < len(char_tokens))
# "th" is the most frequent adjacent pair in the corpus, so it must be merged.
learned_th = ("t", "h") in merges

print("")
print(f"STEP 3: round-trips exactly      : {round_trips}")
print(f"        fewer tokens than chars  : {compresses} ({len(char_tokens)} -> {len(bpe_tokens)})")
print(f"        learned the 'th' merge   : {learned_th}")

ok = round_trips and compresses and learned_th
print("")
print(f"BPE TOKENIZER ROUND-TRIPS AND MERGES CORRECTLY: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("You can now explain how a model sees text. Next round: attention.")
