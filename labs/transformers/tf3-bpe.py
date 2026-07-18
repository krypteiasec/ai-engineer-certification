#!/usr/bin/env python3
"""
LAB TF3: Byte-pair encoding (BPE), the tokenizer real models actually use.

Course 1 used a character tokenizer: one character, one token. That is simple but
wasteful, because common sequences like "th" or "ing" cost a token each every
time. BPE fixes this by LEARNING its vocabulary from data. Start with characters,
then repeatedly find the most frequent adjacent pair and MERGE it into a new
single token. Do that N times and the frequent chunks of your language become
their own tokens. GPT-2, Llama, and friends all tokenize this way (over bytes).

You build BPE training from scratch here: count pairs, merge the top pair, repeat.
Then you PROVE the two properties that matter:
  - the merges cut the token COUNT (the corpus encodes into fewer tokens),
  - encode then decode is LOSSLESS (you get the exact text back).

Run: python3 modules/academy-content/labs/transformers/tf3-bpe.py
"""
import sys
from collections import Counter

# A tiny corpus with obvious repeated structure BPE should discover.
CORPUS = "the theme of the theory is the thesis the theatre theme"
NUM_MERGES = 8


def get_pairs(tokens):
    """Count every adjacent token pair in the sequence."""
    return Counter(zip(tokens, tokens[1:]))


def merge(tokens, pair, new_tok):
    """Replace every occurrence of the adjacent `pair` with `new_tok`."""
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_tok)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


# STEP 1: start from characters. Each character is its own token.
tokens = list(CORPUS)
start_len = len(tokens)
print("STEP 1: char-level start: %d tokens, %d unique" % (start_len, len(set(tokens))))

# STEP 2: learn merges. Each round, merge the single most frequent adjacent pair.
merges = []          # ordered list of learned merge rules
vocab = set(tokens)  # grows by one per merge
print("STEP 2: learning %d merges (most frequent adjacent pair each round):" % NUM_MERGES)
for step in range(NUM_MERGES):
    pairs = get_pairs(tokens)
    if not pairs:
        break
    # deterministic: break ties by the pair itself so runs are reproducible.
    best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
    new_tok = best[0] + best[1]         # the merged string becomes the new token
    tokens = merge(tokens, best, new_tok)
    merges.append(best)
    vocab.add(new_tok)
    print("   merge %d: %-14r count=%d  ->  token %r" % (step + 1, "".join(best), pairs[best], new_tok))

end_len = len(tokens)
print("STEP 3: after merges: %d tokens (was %d), vocab grew to %d" % (end_len, start_len, len(vocab)))


def encode(text):
    """Apply the LEARNED merges, in order, to fresh text."""
    toks = list(text)
    for a, b in merges:
        toks = merge(toks, (a, b), a + b)
    return toks


def decode(toks):
    """Merged tokens are just their own substrings, so joining reverses encoding."""
    return "".join(toks)


# STEP 4: the invariants.
#  (a) fewer tokens: learned merges compress the corpus below char-level count.
enc = encode(CORPUS)
fewer = len(enc) < start_len
#  (b) lossless: decode(encode(text)) == text, on the corpus AND on unseen text.
rt_corpus = decode(encode(CORPUS)) == CORPUS
unseen = "the theory theme"
rt_unseen = decode(encode(unseen)) == unseen
lossless = rt_corpus and rt_unseen
print("STEP 4: encoded corpus is %d tokens vs %d char tokens (fewer): %s" % (
    len(enc), start_len, "YES" if fewer else "NO"))
print("        round-trip lossless on corpus and unseen text: %s" % ("YES" if lossless else "NO"))

ok = fewer and lossless and len(vocab) == len(set(list(CORPUS))) + len(merges)
print("")
print("BPE MERGES CUT TOKEN COUNT AND ROUND-TRIP IS LOSSLESS: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("")
print("You learned a vocabulary from data. Frequent chunks became single tokens,")
print("the sequence got shorter, and nothing was lost. That is BPE.")
