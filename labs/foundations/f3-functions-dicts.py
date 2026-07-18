#!/usr/bin/env python3
"""
LAB F3: Functions and dictionaries. Reusable logic and lookups.

A function packages logic you can reuse. A dictionary maps keys to values, which
is exactly how a tokenizer maps characters to ids. Here you write a function that
counts how often each character appears, which is your first step toward the
counting you will do in the LLM course.

Run: python3 modules/academy-content/labs/foundations/f3-functions-dicts.py
"""
import sys

# STEP 1: a function takes inputs and returns an output.
def char_counts(text):
    counts = {}                       # a dictionary: char -> how many times seen
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return counts

text = "the cat sat"
counts = char_counts(text)
print("STEP 1-2: count characters with a function and a dictionary")
for ch in sorted(counts):
    shown = "space" if ch == " " else ch
    print(f"  {shown!r:8} -> {counts[ch]}")

# STEP 3: invariants. The counts must sum to the length of the text (nothing lost
# or double counted), and looking up a known char returns the right number.
total = sum(counts.values())
ok = (total == len(text)) and (counts["t"] == 3) and (counts[" "] == 2)
print("")
print(f"STEP 3: counts sum to the text length and match by hand: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("You just built a character counter, the heart of a bigram model. Next: vectors.")
