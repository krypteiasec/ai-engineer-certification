#!/usr/bin/env python3
"""
LAB LO3: Parse a TELOS file and extract the goals.

TELOS is your machine-readable life: missions, goals, problems, strategies,
narratives, challenges. It is plain markdown on purpose, so any tool (including
your DA, and this lab) can read it with nothing but text parsing. Here you parse
a SAMPLE TELOS for a fictional user, pull out every mission and goal, and prove
each one has a stable ID. The sample below is invented for the lesson; it is not
anyone's real TELOS.

Run: python3 modules/academy-content/labs/lifeos/lo3-telos-parse.py
"""
import sys, re

# STEP 1: a sample TELOS in the real on-disk shape. Fully fictional content.
SAMPLE_TELOS = """
## Missions
- **M0**: Teach a thousand people to build with AI.
- **M1**: Reach financial independence through owned software.

## Active Goals
- **G0**: Ship the MVP by Q2 with 100 daily users.
- **G1**: Publish 24 essays this year, one every other week.
- **G2**: Bank a six month cash runway.

## Problems Being Solved
- **P0**: Most people fight their tools instead of doing the work.
"""

# STEP 2: parse lines of the form "- **ID**: text" under a section heading.
def parse_items(text, prefix):
    items = []
    pat = re.compile(r"^- \*\*(" + prefix + r"\d+)\*\*:\s*(.+)$")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            items.append((m.group(1), m.group(2)))
    return items

missions = parse_items(SAMPLE_TELOS, "M")
goals = parse_items(SAMPLE_TELOS, "G")
print("STEP 1-2: extracted missions and goals from the sample TELOS")
for gid, txt in goals:
    print(f"  {gid}: {txt}")

# STEP 3: invariants. we found the expected counts, and every goal has a valid
# ID of the form G<number>, which is what makes TELOS entries addressable.
counts_ok = (len(missions) == 2 and len(goals) == 3)
ids_ok = all(re.fullmatch(r"G\d+", gid) for gid, _ in goals)
print("")
print(f"STEP 3: found {len(missions)} missions and {len(goals)} goals; every goal ID valid: {ids_ok}")

ok = counts_ok and ids_ok
print("")
print(f"TELOS PARSED AND GOALS ADDRESSABLE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Plain text means anything can read your goals. Next: how the Algorithm acts on them.")
