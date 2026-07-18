#!/usr/bin/env python3
"""
LAB LM7: Migrate your context. Classify each chunk, route it, keep the provenance.

You do not start LifeOS empty. You already have a life written down somewhere:
old notes, a resume, opinions, a writing sample. The Migrate skill brings that in
by classifying each chunk against the USER taxonomy and committing it to the right
file WITH provenance, a record of where it came from, so nothing is orphaned and
you can always trace a line back to its source. In this lab you route a handful of
sample chunks into the taxonomy and prove each landed in the correct target and
carried its provenance. Every chunk here is invented for the lesson.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm7-migrate-classify.py
"""
import sys

# STEP 1: the USER taxonomy, each target with keywords that route a chunk to it.
TAXONOMY = {
    "TELOS/GOALS.md":       ["goal", "ship", "by 2026", "target", "milestone"],
    "WRITINGSTYLE.md":      ["i write", "voice", "no em dash", "sentence", "tone"],
    "OPINIONS.md":          ["i believe", "i think", "in my view", "opinion"],
    "RESUME.md":            ["experience", "role", "director", "worked at", "years"],
    "CONTACTS.md":          ["email", "phone", "reach", "contact"],
}

def classify(chunk):
    text = chunk.lower()
    best, score = None, 0
    for target, kws in TAXONOMY.items():
        hits = sum(1 for k in kws if k in text)
        if hits > score:
            best, score = target, hits
    return best

# STEP 2: sample chunks, each with its true home (the ground truth) and a source.
chunks = [
    ("Ship the personal-site rebuild by 2026 as my next milestone", "TELOS/GOALS.md", "old-notes.md:12"),
    ("I write in short direct sentences and I never use an em dash", "WRITINGSTYLE.md", "style.txt:3"),
    ("I believe most AI advice is folklore, in my view you test it", "OPINIONS.md", "journal.md:88"),
    ("15 years of experience, most recently a Director role", "RESUME.md", "resume.pdf:1"),
    ("Best email to reach me and my phone are below", "CONTACTS.md", "vcard.txt:1"),
]

print("STEP 1-2: classify and route each chunk, attaching provenance")
routed = []
for text, expected, source in chunks:
    target = classify(text)
    routed.append({"target": target, "expected": expected, "source": source, "committed": target is not None})
    mark = "OK " if target == expected else "!! "
    print(f"  [{mark}] {text[:44]:44} -> {target}   (from {source})")

# STEP 3: invariants. Every chunk routed to its correct taxonomy target, and every
# committed chunk carries a non-empty provenance (source) so none is orphaned.
all_correct = all(r["target"] == r["expected"] for r in routed)
all_have_provenance = all(r["source"] for r in routed if r["committed"])
none_orphaned = all(r["committed"] for r in routed)
print("")
print(f"STEP 3: every chunk routed correctly    : {all_correct}")
print(f"        every commit carries provenance : {all_have_provenance}")
print(f"        nothing orphaned                : {none_orphaned}")

ok = all_correct and all_have_provenance and none_orphaned
print("")
print(f"MIGRATION ROUTES CORRECTLY (every chunk to its taxonomy target, with provenance): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Your existing context is now inside LifeOS, traceable to its source. Next: run it as a daily OS.")
